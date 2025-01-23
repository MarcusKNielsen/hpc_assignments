TARGET_V1 = poisson6.1
OBJS_V1	= main.o \
        jacobi.o \
        alloc3d.o \
        print.o \

TARGET_V6 = poisson6.2
OBJS_V6	= main_part6_2.o \
        jacobi_part6_2.o \
        alloc3d.o \
        alloc3d_dev.o \
        print.o \

TARGET_V7 = poisson7
OBJS_V7	= main_part7.o \
        jacobi_part7.o \
        alloc3d.o \
        alloc3d_dev.o \
        print.o \

CC	= nvc
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=mem:separate:pinnedalloc -gpu=cc90 -gpu=lineinfo -cuda -mp=noautopar
ISA	=
PARA	=
INC   =
LIBS	=

CFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)
CXXFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET_V1) $(TARGET_V6) $(TARGET_V7)

$(TARGET_V1): $(OBJS_V1)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_V1) $(LIBS)

$(TARGET_V6): $(OBJS_V6)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_V6) $(LIBS)

$(TARGET_V7): $(OBJS_V7)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_V7) $(LIBS)

clean:
	@/bin/rm -f *.o $(TARGET_V1) $(TARGET_V6) $(TARGET_V7)
