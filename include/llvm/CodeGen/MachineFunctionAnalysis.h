//===-- MachineFunctionAnalysis.h - Owner of MachineFunctions ----*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachineFunctionAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFUNCTIONANALYSIS_H
#define LLVM_CODEGEN_MACHINEFUNCTIONANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class MachineFunction;
class MachineFunctionInitializer;
class TargetMachine;

/// MachineFunctionAnalysis - This class is a Pass that manages a
/// MachineFunction object.
struct MachineFunctionAnalysis : public FunctionPass {
private:
  const TargetMachine &TM;
  MachineFunction *MF;
  unsigned NextFnNum;
  MachineFunctionInitializer *MFInitializer;
  DenseMap<const Function *, const MachineFunction *> MFDatabase;

public:
  static char ID;
  explicit MachineFunctionAnalysis(const TargetMachine &tm,
                                   MachineFunctionInitializer *MFInitializer);

  MachineFunction &getMF() const { return *MF; }

  /// Lookup \p the MachineFunction of \p F managed by the same instance of
  /// this pass. If \p is not yet transformed to a MachineForm or not
  /// processed by the pass instance, this method returns nullptr.
  const MachineFunction *getMFOf(const Function *F) const {
    return MFDatabase.lookup(F);
  }

  const char* getPassName() const override {
    return "Machine Function Analysis";
  }

private:
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
  void releaseMemory() override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // End llvm namespace

#endif
