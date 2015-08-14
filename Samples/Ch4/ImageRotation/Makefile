# C compiler to use
CC = gcc

# compile-time flags
CFLAGS = -Wall -g

# header file directories
INCLUDES = -I/usr/include -I/opt/AMDAPPSDK-2.9-1/include -I../../Utils

# library paths
LFLAGS = -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -L/usr/local/lib

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = -lbmp -lOpenCL

# define the C source files
SRCS = image-rotation.c ../../Utils/utils.c ../../Utils/bmp-utils.c

# define the C object files 
OBJS = $(SRCS:.c=.o)

# define the executable file 
MAIN = image-rotation 

.PHONY: depend clean

all:    $(MAIN)

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^


