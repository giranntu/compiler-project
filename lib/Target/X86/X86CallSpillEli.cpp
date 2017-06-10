#include "X86.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

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

};

char CallSpillEli::ID = 0;
} // namespace

FunctionPass *llvm::createX86CallSpillEliPass() { return new CallSpillEli(); }

bool CallSpillEli::runOnMachineFunction(MachineFunction &MF) {
  return false;
}


