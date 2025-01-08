TARGET	= libmatmult.so
LIBSRCS	= mult.c
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT     = -g
PIC	= -fPIC

CC      = gcc
CFLAGS  = $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= -lopenblas

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 
