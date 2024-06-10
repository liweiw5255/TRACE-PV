CC 		= mpic++
C		= cpp
DIR 	= ${matlabroot}/extern/bin/glnxa64
CFLAGS 	= -std=c++11 -I${matlabroot}/extern/include -L${DIR}
LFLAGS 	= -lMatlabDataArray -lMatlabEngine

$(if $(matlabroot),,$(error matlabroot is not defined))

TARGET = main

OBJECTS = ${TARGET}.o simulation.o clusterMerging.o

%.o: %.cpp
	${CC} ${CFLAGS} *.${C} 

${TARGET}: ${TARGET}.${C} simulation.${C} clusterMerging.${C}
	${CC} ${CFLAGS} -o ${TARGET} ${TARGET}.${C} simulation.${C} clusterMerging.${C} ${LFLAGS}

#${TARGET}:${TARGET}.o simulation.o 
#	${CC} ${CFLAGS} ${LFLAGS} -o ${TARGET} ${OBJECTS} 

#install:install_name_tool -add_rpath ${DIR} ${TARGET}

run:
	mpirun -np ${CORES} ${TARGET}

clean:
	rm -f ${TARGET} ${TARGET}.o
