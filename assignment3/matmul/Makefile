TARGET= libmatmult.so
OBJS	= mkn_mnk_offload.o matmult_cublas.o matmult_mkn_omp.o blk_offload.o matmult_asy_offload.o

CC	= nvc
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -mp=gpu -gpu=mem:separate:pinnedalloc -gpu=lineinfo -gpu=cc90 -cuda -mp=noautopar -cudalib=cublas
PIC   = -fpic -shared
ISA	=
PARA	=
INC   =
LIBS	= -lopenblas

CXXFLAGS = $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	@/bin/rm -f $(TARGET) $(OBJS)