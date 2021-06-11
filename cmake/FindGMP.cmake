# Try to find the GNU Multiple Precision Arithmetic Library (GMP)
# See http://gmplib.org/

# Once done this will define
#
#  GMP_FOUND - system has GMP lib with correct version
#  GMP_INCLUDE_DIRS - the GMP include directories
#  GMP_LIBRARIES - the GMP libraries

if(GMP_INCLUDE_DIRS AND GMP_LIBRARIES)
    set(GMP_FIND_QUIETLY TRUE)
endif()

find_path(GMP_INCLUDE_DIR NAMES gmp.h PATHS $ENV{GMPDIR} ${INCLUDE_INSTALL_DIR})

find_library(GMP_LIBRARY gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})
find_library(GMPXX_LIBRARY gmpxx PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_INCLUDE_DIR GMP_LIBRARY GMPXX_LIBRARY)
mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY GMPXX_LIBRARY)

set(GMP_INCLUDE_DIRS "${GMP_INCLUDE_DIR}")
set(GMP_LIBRARIES "${GMPXX_LIBRARY}" "${GMP_LIBRARY}")
