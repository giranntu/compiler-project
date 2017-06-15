#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SparseSet.h"
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

  /// To check if specified register is a callee-saved register.
  bool isCalleeSavedReg(const MachineFunction &, unsigned Reg) const;

  /// Check wheter the specified function does not call any children
  /// function.
  bool hasNoCall(const MachineFunction &) const;

  /// A cache data structure used by isCalleeSavedReg.
  mutable struct {
    const MCPhysReg *Ptr;
    SparseSet<unsigned> Regs;
  } CalleeSavedRegsCache;
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
  MachineRegisterInfo &MFRegInfo = MF.getRegInfo();
  const TargetRegisterInfo *RegInfo = MFRegInfo.getTargetRegisterInfo();

  // The utility 'computeRegisterLiveness' can give information about register
  // liveness. It performs linear search on previous and next few instructions
  // of the specified instruction, so there is chance to get uncertain replies.
  // Adjust the last (hidden) argument of this utility to give more chance to
  // find dead registers.
  auto isDeadInCaller = [=](unsigned Reg) -> bool {
    return TheCaller->getOperand(1).clobbersPhysReg(Reg) ||
           CallerBB->computeRegisterLiveness(RegInfo, Reg, TheCaller, 100) ==
               MachineBasicBlock::LQR_Dead;
  };

  // To check if a register is fully free; that is, all its subregisters are
  // not referenced in the machine function.
  auto isFreeReg = [&](unsigned Reg) -> bool {
    auto SubRegIt = MCSubRegIterator(Reg, RegInfo, true);
    while (SubRegIt.isValid()) {
      if (!MFRegInfo.reg_empty(*SubRegIt))
        return false;

      ++SubRegIt;
    }

    return true;
  };

  // To check if two registers have same addressed ability. For example,
  // <RAX> can be addressed by <RAX>, <EAX>, <AX>, <AH>, <AL>, but
  // <RBP> can only be addressed by <RBP>, <EBP>, <BP>, <BPL>.
  auto isSameAddressable = [&](unsigned FromReg, unsigned ToReg) -> bool {
    // Both registers have same addressed ability if FromReg does not have
    // more addressed ability than ToReg.
    if (!X86::GR64_ABCDRegClass.contains(FromReg) ||
        X86::GR64_ABCDRegClass.contains(ToReg)) {
      return true;
    }

    // Although FromReg has more addressed ability than ToReg, there is
    // chance that FromReg is not addressed specially in the machine
    // function.
    unsigned SpecialAddressedReg = RegInfo->getSubReg(FromReg, 2);
    assert(X86::GR8_ABCD_HRegClass.contains(SpecialAddressedReg));
    return MFRegInfo.reg_empty(SpecialAddressedReg);
  };

  bool NoCall = hasNoCall(MF);
  auto wontBeClobbered = [&](unsigned Reg) -> bool {
    return NoCall || isCalleeSavedReg(MF, Reg);
  };

  bool changed = false;
  for (CalleeSavedInstr &CSI : CSInstrs) {
    unsigned SavedReg = CSI.Info.getReg();

    // If SavedReg is originally dead in the caller, there is no longer need
    // to save it.
    if (isDeadInCaller(SavedReg)) {
      DEBUG(dbgs() << "Callee-saved register <" << RegInfo->getName(SavedReg)
                   << "> is discarded to spill in function '" << MF.getName()
                   << "' because it is originally dead in the caller.\n");
      goto DiscardSaving;
    }

    // It is uncertain that SavedReg is dead in the caller, so try to rename
    // the register to another available (non-reserved, not used in callee,
    // and dead in the caller) GPR to look for chance of not saving.
    // Notice that it can only rename to other unused callee-saved register;
    // otherwise, it may be overriden(clobbered) in child functions.
    for (auto OtherReg : reverse(X86::GR64RegClass)) {
      if (!MFRegInfo.isReserved(OtherReg) && wontBeClobbered(OtherReg) &&
          isFreeReg(OtherReg) && isSameAddressable(SavedReg, OtherReg) &&
          isDeadInCaller(OtherReg)) {
        // Rename the register and all its sub-registers.
        MFRegInfo.replaceRegWith(SavedReg, OtherReg);

        auto SubRegIt = MCSubRegIndexIterator(SavedReg, RegInfo);
        while (SubRegIt.isValid()) {
          MFRegInfo.replaceRegWith(
              SubRegIt.getSubReg(),
              RegInfo->getSubReg(OtherReg, SubRegIt.getSubRegIndex()));
          ++SubRegIt;
        }

        CSI.Info = CalleeSavedInfo(OtherReg, CSI.Info.getFrameIdx());
        ++NumRegRenamed;

        DEBUG(dbgs() << "Callee-saved register <" << RegInfo->getName(SavedReg)
                     << "> in function '" << MF.getName() << "' has been "
                     << "renamed to <" << RegInfo->getName(OtherReg) << ">, "
                     << "which is dead in the caller, and the register saving "
                     << "for it is discarded safely.\n");
        goto DiscardSaving;
      }
    }

    // There is no chance to discard the register saving in the callee using
    // current techniques.
    DEBUG(dbgs() << "Callee-saved register <" << RegInfo->getName(SavedReg)
                 << "> in function '" << MF.getName() << "' is NOT discarded "
                 << "because it as well as other registers it can rename to "
                 << "are not sure to be originally dead in the caller.\n");
    continue;

DiscardSaving:
    discardRegSave(CSI);
    changed = true;
  }

  return changed;
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
      assert((MInstr->getOpcode() == X86::PUSH64r ||
              MInstr->getOpcode() == X86::POP64r) &&
             MInstr->getOperand(0).getReg() == CSI.Info.getReg());

      MInstr->eraseFromParent();
      ++NumMInstEliminated;
    }
  }
}

bool CallSpillEli::isCalleeSavedReg(const MachineFunction &MF,
                                    unsigned Reg) const {
  const TargetRegisterInfo *RegInfo = MF.getRegInfo().getTargetRegisterInfo();
  const MCPhysReg *Ptr = RegInfo->getCalleeSavedRegs(&MF);

  if (Ptr != CalleeSavedRegsCache.Ptr) {
    CalleeSavedRegsCache.Ptr = Ptr;
    CalleeSavedRegsCache.Regs.clear();
    CalleeSavedRegsCache.Regs.setUniverse(RegInfo->getNumRegs() + 1);
    while (*Ptr) {
      CalleeSavedRegsCache.Regs.insert(*Ptr);
      ++Ptr;
    }
  }

  return CalleeSavedRegsCache.Regs.count(Reg);
}

bool CallSpillEli::hasNoCall(const MachineFunction &MF) const {
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isCall())
        return false;
    }
  }

  return true;
}
