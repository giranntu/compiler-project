#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
using namespace llvm;
using namespace std;

#define DEBUG_TYPE "x86-eliminate-call-spill"

STATISTIC(NumSpillEliminated, "Number of register spills eliminated");
STATISTIC(NumMInstEliminated, "Number of instructions eliminated");
STATISTIC(NumRegRenamed, "Number of register renamed in callee");
STATISTIC(NumVersionAdded, "Number of additional versions of functions");

namespace {
/// A pass to reorder functions in a module to make it the DFS
/// traversal order of the call graph.
struct FunctionReorderPass : public ModulePass {
  static char ID;
  FunctionReorderPass() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<CallGraphWrapperPass>();
  }

  bool runOnModule(Module &) override;
};

struct CallSpillEli : public MachineFunctionPass {
  static char ID;
  CallSpillEli() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "X86 call spillings eliminator";
  }

private:
  /// A structure to record push/pop instructions in prologues
  /// and epilogues of each callee saved register.
  struct CalleeSavedInstr {
    CalleeSavedInfo Info;
    SmallVector<MachineInstr *, 4> PushInstrs;
    SmallVector<MachineInstr *, 4> PopInstrs;
  };

  /// Find save and restore blocks in the machine function.
  void findSaveRestoreBBs(MachineFunction &,
                          SmallVectorImpl<MachineBasicBlock *> &SaveBBs,
                          SmallVectorImpl<MachineBasicBlock *> &RestoreBBs) const;

  /// To find push/pop instructions for each callee saved instruction.
  void findCSInstrs(MachineFunction &,
                    const SmallVectorImpl<MachineBasicBlock *> &SaveBBs,
                    const SmallVectorImpl<MachineBasicBlock *> &RestoreBBs,
                    vector<CalleeSavedInstr> &) const;

  /// Find callers of the machine function <Of>. The result machine instructions
  /// are stored in <Callers>. It returns false if the MachineFunction of any
  /// caller is still unavailable.
  bool findCallers(const MachineFunction &Of,
                   const MachineFunctionAnalysis &,
                   SmallVectorImpl<const MachineInstr *> &Callers) const;

  /// Discard saving/restoring the specified callee-saved register.
  void discardRegSave(CalleeSavedInstr &) const;
};

char FunctionReorderPass::ID = 0;
char CallSpillEli::ID = 0;
} // namespace

ModulePass *llvm::createTopdownFunctionReorderPass() { return new FunctionReorderPass(); }
FunctionPass *llvm::createX86CallSpillEliPass() { return new CallSpillEli(); }

bool FunctionReorderPass::runOnModule(Module &M) {
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  Module::FunctionListType &FunctionList = M.getFunctionList();

  // Sort functions in DFS order. Instead of directly modifying the
  // function list, the result order is stored in a vector because
  // the modification may break the loop iteration.
  vector<Function *> FunctionOrder;
  FunctionOrder.reserve(FunctionList.size());
  for (auto Node : depth_first(&CG)) {
    // Don't count external functions, which are those returning nullptr.
    if (Function *F = Node->getFunction()) {
      FunctionOrder.push_back(F);
    }
  }
  assert(FunctionOrder.size() == FunctionList.size());

  // Adjust order of functions in FunctionList.
  for (Function *F : FunctionOrder) {
    FunctionList.remove(F);
    FunctionList.push_back(F);
  }

  return true;
}

bool CallSpillEli::runOnMachineFunction(MachineFunction &MF) {
  MachineFunctionAnalysis &MFA = getAnalysis<MachineFunctionAnalysis>();
  const Function *F = MF.getFunction();

  // Require callers of the function to be determinable, and abort if not.
  // Thatis, it cannot have either external linkage (may be called from
  // other modules) or address taken (may be called indirectly).
  if (F->hasAddressTaken() || !F->hasLocalLinkage()) {
    DEBUG(dbgs() << "Callers of function '" << MF.getName() << "' are not "
                 << "trackable because either it has external linkage or it "
                 << "might be called indirectly (it has its address taken). "
                 << "Aborted.\n");
    return false;
  }

  // Require the function to has only one caller. Currently abort if not.
  // TODO: It shall be able to analysis register usage of all callers.
  if (F->getNumUses() != 1) {
    DEBUG(dbgs() << "Function '" << MF.getName() << "' has either zero or "
                 << "more than one callers which is currently not supported. "
                 << "Aborted.\n");
    return false;
  }

  // Find all callers to this MachineFunction.
  // Require all callers' register allocation to be done. Abort if not.
  SmallVector<const MachineInstr *, 4> Callers;
  if (!findCallers(MF, MFA, Callers)) {
    DEBUG(dbgs() << "Register allocation for some callers of function '"
                 << MF.getName() << "' are not yet done. Aborted.\n");
    return false;
  }

  SmallVector<MachineBasicBlock *, 4> SaveBBs, RestoreBBs;
  vector<CalleeSavedInstr> CSInstrs;

  // Find original save/restore instructions.
  findSaveRestoreBBs(MF, SaveBBs, RestoreBBs);
  assert(!SaveBBs.empty() && !RestoreBBs.empty());
  findCSInstrs(MF, SaveBBs, RestoreBBs, CSInstrs);
  if (CSInstrs.empty()) {
    DEBUG(dbgs() << "Function '" << MF.getName() << "' does not save/restore "
                 << "any callee-saved register. Aborted.\n");
    return false;
  }

  // TODO: Currently it only supports one caller. However, following code
  // need modification to support multiple callers.
  const MachineInstr *TheCaller = Callers.front();
  const MachineBasicBlock *CallerBB = TheCaller->getParent();
  const TargetRegisterInfo *RegInfo = MF.getRegInfo().getTargetRegisterInfo();

  for (CalleeSavedInstr &CSI : CSInstrs) {
    // The utility 'computeRegisterLiveness' can give information about register
    // liveness. It performs linear search on previous and next few instructions
    // of the specified instruction, so there is chance to get uncertain replies.
    // Adjust the last (hidden) argument of this utility to give more chance to
    // find dead registers.
    if (CallerBB->computeRegisterLiveness(RegInfo, CSI.Info.getReg(), TheCaller) ==
        MachineBasicBlock::LQR_Dead) {
      DEBUG(dbgs() << "Callee-saved register <" << RegInfo->getName(CSI.Info.getReg())
                   << "> is discarded to spill in function '" << MF.getName()
                   << "' because it is originally dead in the caller.\n");
      discardRegSave(CSI);
    }
  }

  return false;
}

/// The implementation comes from PEI::calculateSaveRestoreBlocks in
/// lib/CodeGen/PrologueEpilogueInserter.cpp
void CallSpillEli::findSaveRestoreBBs(
    MachineFunction &MF, SmallVectorImpl<MachineBasicBlock *> &SaveBBs,
    SmallVectorImpl<MachineBasicBlock *> &RestoreBBs) const {
  SaveBBs.clear();
  RestoreBBs.clear();
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // Even when we do not change any CSR, we still want to insert the
  // prologue and epilogue of the function.
  // So set the save points for those.

  // Use the points found by shrink-wrapping, if any.
  if (MFI->getSavePoint()) {
    SaveBBs.push_back(MFI->getSavePoint());
    assert(MFI->getRestorePoint() && "Both restore and save must be set");
    MachineBasicBlock *RestoreBlock = MFI->getRestorePoint();
    // If RestoreBlock does not have any successor and is not a return block
    // then the end point is unreachable and we do not need to insert any
    // epilogue.
    if (!RestoreBlock->succ_empty() || RestoreBlock->isReturnBlock())
      RestoreBBs.push_back(RestoreBlock);
    return;
  }

  // Save refs to entry and return blocks.
  SaveBBs.push_back(&MF.front());
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.isEHFuncletEntry())
      SaveBBs.push_back(&MBB);
    if (MBB.isReturnBlock())
      RestoreBBs.push_back(&MBB);
  }
}

void CallSpillEli::findCSInstrs(
    MachineFunction &MF, const SmallVectorImpl<MachineBasicBlock *> &SaveBBs,
    const SmallVectorImpl<MachineBasicBlock *> &RestoreBBs,
    vector<CalleeSavedInstr> &CSInstrs) const {
  MachineFrameInfo &MFI = *MF.getFrameInfo();

  SmallVector<MachineBasicBlock::iterator, 4> SaveBBIts(SaveBBs.size());
  SmallVector<MachineBasicBlock::reverse_iterator, 4> RestoreBBIts(RestoreBBs.size());
  transform(SaveBBs, SaveBBIts.begin(), [](MachineBasicBlock *MBB) { return MBB->begin(); });
  transform(RestoreBBs, RestoreBBIts.begin(), [](MachineBasicBlock *MBB) { return MBB->rbegin(); });

  vector<CalleeSavedInfo> CSIs(MFI.getCalleeSavedInfo());
  sort(CSIs.begin(), CSIs.end(), [&](const CalleeSavedInfo &lhs, const CalleeSavedInfo &rhs) {
    return MFI.getObjectOffset(lhs.getFrameIdx()) > MFI.getObjectOffset(rhs.getFrameIdx());
  });

  CSInstrs.clear();
  for (auto &CSI : CSIs) {
    CalleeSavedInstr CSInstr {CSI};

    // Find pushes in prologues.
    for (auto &SaveBBIt : SaveBBIts) {
      while (SaveBBIt->getOpcode() != X86::PUSH64r ||
             SaveBBIt->getOperand(0).getReg() != CSI.getReg()) {
        ++SaveBBIt;
      }
      assert(SaveBBIt->getFlag(MachineInstr::FrameSetup));
      CSInstr.PushInstrs.push_back(&*SaveBBIt);
    }

    // Find pops in epilogues.
    for (auto &RestoreBBIt : RestoreBBIts) {
      while (RestoreBBIt->getOpcode() != X86::POP64r ||
             RestoreBBIt->getOperand(0).getReg() != CSI.getReg()) {
        ++RestoreBBIt;
      }
      assert(RestoreBBIt->getFlag(MachineInstr::FrameDestroy));
      CSInstr.PopInstrs.push_back(&*RestoreBBIt);
    }

    CSInstrs.emplace_back(move(CSInstr));
  }
}

bool CallSpillEli::findCallers(
    const MachineFunction &MF, const MachineFunctionAnalysis &MFAnalysis,
    SmallVectorImpl<const MachineInstr *> &Results) const {
  DenseSet<const Function *> Searched;
  const Function *F = MF.getFunction();

  Results.clear();
  for (const User *U : F->users()) {
    const Function *CallerFn = cast<Instruction>(U)->getFunction();

    // There may be multiple callers in one function.
    // Skip searched functions.
    if (!Searched.insert(CallerFn).second) {
      continue;
    }

    if (const MachineFunction *MF = MFAnalysis.getMFOf(CallerFn)) {
      for (auto &MBB : *MF) {
        for (auto &MInstr : MBB) {
          if (MInstr.isCall() && MInstr.getOperand(0).getGlobal() == F) {
            Results.push_back(&MInstr);
          }
        }
      }
    } else {
      return false;
    }
  }

  assert(Results.size() == F->getNumUses());
  return true;
}

void CallSpillEli::discardRegSave(CalleeSavedInstr &CSI) const {
  ++NumSpillEliminated;

  for (auto &PushPops : { CSI.PushInstrs, CSI.PopInstrs }) {
    for (MachineInstr *MInstr : PushPops) {
      MInstr->eraseFromParent();
      ++NumMInstEliminated;
    }
  }
}
