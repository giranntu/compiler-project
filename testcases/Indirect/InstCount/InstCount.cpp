#include <iomanip>
#include <iostream>
#include <string.h>
#include "pin.H"

// Holds instruction count for a single procedure
typedef struct RtnInfo
{
  string _name;
  UINT64 _rtnCount;
  UINT64 _icount;
  UINT64 _itotal;
  UINT64 _iself;
  struct RtnInfo * _cur_caller;
  struct RtnInfo * _next;
} RTN_INFO;


// Last routine called
RtnInfo * lastRtn = 0;

// Linked list of instruction counts for each routine
RTN_INFO * RtnList = 0;

// Program path
string prog_path;


VOID doinst(RTN_INFO * r) {
  r->_icount++;
  r->_iself++;
}

VOID rtn_called(RTN_INFO * r) {
  r->_rtnCount++;
  r->_cur_caller = lastRtn;
  lastRtn = r;
}

VOID rtn_returned(RTN_INFO * r) {
  // this function is called before the RET instruction, so add the count by 1
  r->_icount ++;
  r->_itotal += r->_icount;
  if (r->_cur_caller) {
    r->_cur_caller->_icount += r->_icount;
  }
  lastRtn = r->_cur_caller;
  // do not count the additional RET next time
  r->_icount = -1;
}

// Pin calls this function every time a new rtn is executed
VOID Routine(RTN rtn, VOID *v) {

  // Allocate a counter for this routine
  RTN_INFO * r = new RTN_INFO;

  // Do not instrument shared library
  if (IMG_Name(SEC_Img(RTN_Sec(rtn)))!=prog_path) 
    return;

  r->_name = RTN_Name(rtn);
  r->_icount = 0;
  r->_itotal = 0;
  r->_iself = 0;
  r->_rtnCount = 0;
  r->_cur_caller = 0;

  // Add to list of routines
  r->_next = RtnList;
  RtnList = r;

  RTN_Open(rtn);

  RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)rtn_called, IARG_PTR, r, IARG_END);
  RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)rtn_returned, IARG_PTR, r, IARG_END);

  // For each instruction of the routine
  for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins)) {
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)doinst, IARG_PTR, r, IARG_END);
  }


  RTN_Close(rtn);
}

// This function is called when the application exits
// It prints the name and count for each procedure
VOID Fini(INT32 code, VOID *v) {
  cerr << setw(23) << "procedure" << " "
      << setw(12) << "calls" << " "
      << setw(12) << "self_insts" << " "
      << setw(12) << "total_insts" << endl;
  for (RTN_INFO * r = RtnList; r; r = r->_next) {
      cerr << setw(23) << r->_name << " "
          << setw(12) << r->_rtnCount << " "
          << setw(12) << r->_iself << " "
          << setw(12) << r->_itotal << endl;
  }
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage() {
  cerr << "This Pintool counts the number of times a routine is executed" << endl;
  cerr << "and the number of instructions executed in a routine" << endl;
  cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
  return -1;
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char * argv[]) {
  // Initialize symbol table code, needed for rtn instrumentation
  PIN_InitSymbols();

  // Initialize pin
  if (PIN_Init(argc, argv)) return Usage();

  // Parse program name
  char buf[PATH_MAX];
  for (int i=0; i<argc; ++i) {
    if (!strcmp(argv[i], "--")) {
      realpath(argv[i+1], buf);
      break;
    }
  }
  prog_path = buf;


  // Register Routine to be called to instrument rtn
  RTN_AddInstrumentFunction(Routine, 0);

  // Register Fini to be called when the application exits
  PIN_AddFiniFunction(Fini, 0);

  // Start the program, never returns
  PIN_StartProgram();

  return 0;
}
