#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
using namespace llvm;
using namespace std;

#define DEBUG_TYPE "x86-eliminate-call-spill"

STATISTIC(NumSpillEliminated, "Number of register spills eliminated");
STATISTIC(NumRegRenamed, "Number of register renamed in callee");
STATISTIC(NumVersionAdded, "Number of additional versions of functions");

namespace {
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
                          SmallVectorImpl<MachineBasicBlock *> &RestoreBBs);

  /// To find push/pop instructions for each callee saved instruction.
  void findCSInstrs(MachineFrameInfo &,
                    const SmallVectorImpl<MachineBasicBlock *> &SaveBBs,
                    const SmallVectorImpl<MachineBasicBlock *> &RestoreBBs,
                    vector<CalleeSavedInstr> &);
};

char CallSpillEli::ID = 0;
} // namespace

FunctionPass *llvm::createX86CallSpillEliPass() { return new CallSpillEli(); }

bool CallSpillEli::runOnMachineFunction(MachineFunction &MF) {
  SmallVector<MachineBasicBlock *, 4> SaveBBs, RestoreBBs;
  vector<CalleeSavedInstr> CSInstrs;

  findSaveRestoreBBs(MF, SaveBBs, RestoreBBs);
  assert(!SaveBBs.empty() && !RestoreBBs.empty());
  findCSInstrs(*MF.getFrameInfo(), SaveBBs, RestoreBBs, CSInstrs);

  return false;
}

/// The implementation comes from PEI::calculateSaveRestoreBlocks in
/// lib/CodeGen/PrologueEpilogueInserter.cpp
void CallSpillEli::findSaveRestoreBBs(MachineFunction &MF,
                                      SmallVectorImpl<MachineBasicBlock *> &SaveBBs,
                                      SmallVectorImpl<MachineBasicBlock *> &RestoreBBs) {
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

void CallSpillEli::findCSInstrs(MachineFrameInfo &MFI,
                                const SmallVectorImpl<MachineBasicBlock *> &SaveBBs,
                                const SmallVectorImpl<MachineBasicBlock *> &RestoreBBs,
                                vector<CalleeSavedInstr> &CSInstrs) {
  SmallVector<MachineBasicBlock::iterator, 4> SaveBBIts(SaveBBs.size());
  SmallVector<MachineBasicBlock::reverse_iterator, 4> RestoreBBIts(RestoreBBs.size());
  transform(SaveBBs, SaveBBIts.begin(), [](MachineBasicBlock *MBB) { return MBB->begin(); });
  transform(RestoreBBs, RestoreBBIts.begin(), [](MachineBasicBlock *MBB) { return MBB->rbegin(); });

  vector<CalleeSavedInfo> CSIs(MFI.getCalleeSavedInfo());
  sort(CSIs.begin(), CSIs.end(), [](const CalleeSavedInfo &lhs, const CalleeSavedInfo &rhs) {
    return lhs.getFrameIdx() > rhs.getFrameIdx();
  });

  CSInstrs.clear();
  for (auto &CSI : CSIs) {
    CalleeSavedInstr CSInstr {CSI};

    // Find pushes in prologues.
    for (auto &SaveBBIt : SaveBBIts) {
      while (SaveBBIt->getOpcode() != X86::PUSH64r)
        ++SaveBBIt;
      assert(SaveBBIt->getFlag(MachineInstr::FrameSetup) &&
             SaveBBIt->getOperand(0).getReg() == CSI.getReg());
      CSInstr.PushInstrs.push_back(&*(SaveBBIt++));
    }

    // Find pops in epilogues.
    for (auto &RestoreBBIt : RestoreBBIts) {
      while (RestoreBBIt->getOpcode() != X86::POP64r)
        ++RestoreBBIt;
      assert(RestoreBBIt->getFlag(MachineInstr::FrameDestroy) &&
             RestoreBBIt->getOperand(0).getReg() == CSI.getReg());
      CSInstr.PopInstrs.push_back(&*(RestoreBBIt++));
    }

    CSInstrs.emplace_back(move(CSInstr));
  }
}
