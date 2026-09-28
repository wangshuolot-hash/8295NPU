// SA8295P GVM vendor libcdsprpc.so workaround.
//
// The vendor QoS control path (fastrpc latency voting, invoked by libQnnHtp
// during context setup and graph finalize/execute) pairs every
// rpcmem_init/rpcmem_deinit call, and rpcmem_deinit closes the process-global
// ION fd. Any rpcmem_alloc issued afterwards (including QNN-internal
// allocations inside qnn_graph_finalize) hits an invalid fd and fails with
// QNN_GRAPH_ERROR_MEM_ALLOC.
//
// Interpose rpcmem_deinit* as no-ops so the global ION fd stays open for the
// whole process lifetime. rpcmem_init stays the real one (it is idempotent
// when the fd is already open).

int rpcmem_deinit(void) { return 0; }

int rpcmem_deinit_global(void) { return 0; }
