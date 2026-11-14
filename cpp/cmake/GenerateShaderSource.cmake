# GenerateShaderSource.cmake
# Converts a binary SPIR-V file to a C source file with embedded byte array.
#
# Required variables:
#   SPV_FILE    - Path to the input .spv file
#   OUT_FILE    - Path to the output .c file
#   SHADER_NAME - Base name for the generated symbols

if(NOT DEFINED SPV_FILE)
  message(FATAL_ERROR "SPV_FILE not defined")
endif()
if(NOT DEFINED OUT_FILE)
  message(FATAL_ERROR "OUT_FILE not defined")
endif()
if(NOT DEFINED SHADER_NAME)
  message(FATAL_ERROR "SHADER_NAME not defined")
endif()

# Read binary file
file(READ "${SPV_FILE}" SPIRV_HEX HEX)
string(LENGTH "${SPIRV_HEX}" SPIRV_HEX_LENGTH)
math(EXPR SPIRV_SIZE "${SPIRV_HEX_LENGTH} / 2")

# Convert hex string to comma-separated byte array
set(BYTE_ARRAY "")
set(LINE_BYTES 0)
math(EXPR LAST_INDEX "${SPIRV_HEX_LENGTH} - 2")

foreach(i RANGE 0 ${LAST_INDEX} 2)
  string(SUBSTRING "${SPIRV_HEX}" ${i} 2 BYTE)
  if(LINE_BYTES EQUAL 0)
    string(APPEND BYTE_ARRAY "  ")
  endif()
  string(APPEND BYTE_ARRAY "0x${BYTE}")
  math(EXPR NEXT_I "${i} + 2")
  if(NEXT_I LESS SPIRV_HEX_LENGTH)
    string(APPEND BYTE_ARRAY ",")
  endif()
  math(EXPR LINE_BYTES "${LINE_BYTES} + 1")
  if(LINE_BYTES EQUAL 16)
    string(APPEND BYTE_ARRAY "\n")
    set(LINE_BYTES 0)
  endif()
endforeach()

# Generate C source file
file(WRITE "${OUT_FILE}"
"/* Auto-generated from ${SHADER_NAME}.spv - DO NOT EDIT */
#include <stddef.h>

const unsigned char _binary_${SHADER_NAME}_start[] = {
${BYTE_ARRAY}
};

const size_t _binary_${SHADER_NAME}_size = ${SPIRV_SIZE};
const unsigned char* _binary_${SHADER_NAME}_end = _binary_${SHADER_NAME}_start + ${SPIRV_SIZE};
")

message(STATUS "Generated ${OUT_FILE} (${SPIRV_SIZE} bytes)")
