#include "M4ARM64Memory.h"
#include "M4DarwinARM64Context.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <signal.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
struct hardware_result
{
	std::uint64_t first, second, base, pc;
	std::array<std::byte, 48> vectors;
};
using vector_pair = std::array<std::byte, 48>;
using probe_type = hardware_result (*)(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, const vector_pair&);
#define ARM_CASE(NAME, INSTRUCTION, BEFORE, AFTER) \
[[gnu::noinline]] hardware_result NAME(std::uint64_t base, std::uint64_t index, std::uint64_t first, std::uint64_t second, const vector_pair& vectors) \
{ \
	hardware_result result{}; \
	asm volatile("mov x2, %[first]\nmov x3, %[second]\nmov x4, %[base]\nmov x5, %[index]\n" \
		"ldp q2, q3, [%[vectors]]\nldr q31, [%[vectors], #32]\n" BEFORE "adr x6, 1f\n1: " INSTRUCTION "\n" AFTER \
		"stp x2, x3, [%[output]]\nstr x4, [%[output], #16]\nstr x6, [%[output], #24]\n" \
		"stp q2, q3, [%[output], #32]\nstr q31, [%[output], #64]\n" \
		: : [first] "r"(first), [second] "r"(second), [base] "r"(base), [index] "r"(index), [vectors] "r"(vectors.data()), [output] "r"(&result) \
		: "x2", "x3", "x4", "x5", "x6", "x7", "v2", "v3", "v31", "memory"); \
	return result; \
}
ARM_CASE(c0, "ldrb w2, [x4, #16]", "", "")
ARM_CASE(c1, "ldurb w2, [x4, #-7]", "", "")
ARM_CASE(c2, "ldrb w2, [x4, #16]!", "", "")
ARM_CASE(c3, "ldrb w2, [x4], #-16", "", "")
ARM_CASE(c4, "ldrb w2, [x4, x5]", "", "")
ARM_CASE(c5, "ldrb w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c6, "ldrb w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c7, "ldrb w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c8, "ldrb w2, [x4, x5, lsl #0]", "", "")
ARM_CASE(c9, "ldrh w2, [x4, #16]", "", "")
ARM_CASE(c10, "ldurh w2, [x4, #-7]", "", "")
ARM_CASE(c11, "ldrh w2, [x4, #16]!", "", "")
ARM_CASE(c12, "ldrh w2, [x4], #-16", "", "")
ARM_CASE(c13, "ldrh w2, [x4, x5]", "", "")
ARM_CASE(c14, "ldrh w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c15, "ldrh w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c16, "ldrh w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c17, "ldrh w2, [x4, x5, lsl #1]", "", "")
ARM_CASE(c18, "ldr w2, [x4, #16]", "", "")
ARM_CASE(c19, "ldur w2, [x4, #-7]", "", "")
ARM_CASE(c20, "ldr w2, [x4, #16]!", "", "")
ARM_CASE(c21, "ldr w2, [x4], #-16", "", "")
ARM_CASE(c22, "ldr w2, [x4, x5]", "", "")
ARM_CASE(c23, "ldr w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c24, "ldr w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c25, "ldr w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c26, "ldr w2, [x4, x5, lsl #2]", "", "")
ARM_CASE(c27, "ldr x2, [x4, #16]", "", "")
ARM_CASE(c28, "ldur x2, [x4, #-7]", "", "")
ARM_CASE(c29, "ldr x2, [x4, #16]!", "", "")
ARM_CASE(c30, "ldr x2, [x4], #-16", "", "")
ARM_CASE(c31, "ldr x2, [x4, x5]", "", "")
ARM_CASE(c32, "ldr x2, [x4, w5, uxtw]", "", "")
ARM_CASE(c33, "ldr x2, [x4, w5, sxtw]", "", "")
ARM_CASE(c34, "ldr x2, [x4, x5, sxtx]", "", "")
ARM_CASE(c35, "ldr x2, [x4, x5, lsl #3]", "", "")
ARM_CASE(c36, "ldrsb w2, [x4, #16]", "", "")
ARM_CASE(c37, "ldursb w2, [x4, #-7]", "", "")
ARM_CASE(c38, "ldrsb w2, [x4, #16]!", "", "")
ARM_CASE(c39, "ldrsb w2, [x4], #-16", "", "")
ARM_CASE(c40, "ldrsb w2, [x4, x5]", "", "")
ARM_CASE(c41, "ldrsb w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c42, "ldrsb w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c43, "ldrsb w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c44, "ldrsb w2, [x4, x5, lsl #0]", "", "")
ARM_CASE(c45, "ldrsb x2, [x4, #16]", "", "")
ARM_CASE(c46, "ldursb x2, [x4, #-7]", "", "")
ARM_CASE(c47, "ldrsb x2, [x4, #16]!", "", "")
ARM_CASE(c48, "ldrsb x2, [x4], #-16", "", "")
ARM_CASE(c49, "ldrsb x2, [x4, x5]", "", "")
ARM_CASE(c50, "ldrsb x2, [x4, w5, uxtw]", "", "")
ARM_CASE(c51, "ldrsb x2, [x4, w5, sxtw]", "", "")
ARM_CASE(c52, "ldrsb x2, [x4, x5, sxtx]", "", "")
ARM_CASE(c53, "ldrsb x2, [x4, x5, lsl #0]", "", "")
ARM_CASE(c54, "ldrsh w2, [x4, #16]", "", "")
ARM_CASE(c55, "ldursh w2, [x4, #-7]", "", "")
ARM_CASE(c56, "ldrsh w2, [x4, #16]!", "", "")
ARM_CASE(c57, "ldrsh w2, [x4], #-16", "", "")
ARM_CASE(c58, "ldrsh w2, [x4, x5]", "", "")
ARM_CASE(c59, "ldrsh w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c60, "ldrsh w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c61, "ldrsh w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c62, "ldrsh w2, [x4, x5, lsl #1]", "", "")
ARM_CASE(c63, "ldrsh x2, [x4, #16]", "", "")
ARM_CASE(c64, "ldursh x2, [x4, #-7]", "", "")
ARM_CASE(c65, "ldrsh x2, [x4, #16]!", "", "")
ARM_CASE(c66, "ldrsh x2, [x4], #-16", "", "")
ARM_CASE(c67, "ldrsh x2, [x4, x5]", "", "")
ARM_CASE(c68, "ldrsh x2, [x4, w5, uxtw]", "", "")
ARM_CASE(c69, "ldrsh x2, [x4, w5, sxtw]", "", "")
ARM_CASE(c70, "ldrsh x2, [x4, x5, sxtx]", "", "")
ARM_CASE(c71, "ldrsh x2, [x4, x5, lsl #1]", "", "")
ARM_CASE(c72, "ldrsw x2, [x4, #16]", "", "")
ARM_CASE(c73, "ldursw x2, [x4, #-7]", "", "")
ARM_CASE(c74, "ldrsw x2, [x4, #16]!", "", "")
ARM_CASE(c75, "ldrsw x2, [x4], #-16", "", "")
ARM_CASE(c76, "ldrsw x2, [x4, x5]", "", "")
ARM_CASE(c77, "ldrsw x2, [x4, w5, uxtw]", "", "")
ARM_CASE(c78, "ldrsw x2, [x4, w5, sxtw]", "", "")
ARM_CASE(c79, "ldrsw x2, [x4, x5, sxtx]", "", "")
ARM_CASE(c80, "ldrsw x2, [x4, x5, lsl #2]", "", "")
ARM_CASE(c81, "strb w2, [x4, #16]", "", "")
ARM_CASE(c82, "sturb w2, [x4, #-7]", "", "")
ARM_CASE(c83, "strb w2, [x4, #16]!", "", "")
ARM_CASE(c84, "strb w2, [x4], #-16", "", "")
ARM_CASE(c85, "strb w2, [x4, x5]", "", "")
ARM_CASE(c86, "strb w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c87, "strb w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c88, "strb w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c89, "strb w2, [x4, x5, lsl #0]", "", "")
ARM_CASE(c90, "strh w2, [x4, #16]", "", "")
ARM_CASE(c91, "sturh w2, [x4, #-7]", "", "")
ARM_CASE(c92, "strh w2, [x4, #16]!", "", "")
ARM_CASE(c93, "strh w2, [x4], #-16", "", "")
ARM_CASE(c94, "strh w2, [x4, x5]", "", "")
ARM_CASE(c95, "strh w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c96, "strh w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c97, "strh w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c98, "strh w2, [x4, x5, lsl #1]", "", "")
ARM_CASE(c99, "str w2, [x4, #16]", "", "")
ARM_CASE(c100, "stur w2, [x4, #-7]", "", "")
ARM_CASE(c101, "str w2, [x4, #16]!", "", "")
ARM_CASE(c102, "str w2, [x4], #-16", "", "")
ARM_CASE(c103, "str w2, [x4, x5]", "", "")
ARM_CASE(c104, "str w2, [x4, w5, uxtw]", "", "")
ARM_CASE(c105, "str w2, [x4, w5, sxtw]", "", "")
ARM_CASE(c106, "str w2, [x4, x5, sxtx]", "", "")
ARM_CASE(c107, "str w2, [x4, x5, lsl #2]", "", "")
ARM_CASE(c108, "str x2, [x4, #16]", "", "")
ARM_CASE(c109, "stur x2, [x4, #-7]", "", "")
ARM_CASE(c110, "str x2, [x4, #16]!", "", "")
ARM_CASE(c111, "str x2, [x4], #-16", "", "")
ARM_CASE(c112, "str x2, [x4, x5]", "", "")
ARM_CASE(c113, "str x2, [x4, w5, uxtw]", "", "")
ARM_CASE(c114, "str x2, [x4, w5, sxtw]", "", "")
ARM_CASE(c115, "str x2, [x4, x5, sxtx]", "", "")
ARM_CASE(c116, "str x2, [x4, x5, lsl #3]", "", "")
ARM_CASE(c117, "ldr b2, [x4, #32]", "", "")
ARM_CASE(c118, "ldur b2, [x4, #-7]", "", "")
ARM_CASE(c119, "ldr b2, [x4, #16]!", "", "")
ARM_CASE(c120, "ldr b2, [x4], #-16", "", "")
ARM_CASE(c121, "ldr b2, [x4, x5, lsl #0]", "", "")
ARM_CASE(c122, "ldr b2, [x4, w5, sxtw #0]", "", "")
ARM_CASE(c123, "ldr h2, [x4, #32]", "", "")
ARM_CASE(c124, "ldur h2, [x4, #-7]", "", "")
ARM_CASE(c125, "ldr h2, [x4, #16]!", "", "")
ARM_CASE(c126, "ldr h2, [x4], #-16", "", "")
ARM_CASE(c127, "ldr h2, [x4, x5, lsl #1]", "", "")
ARM_CASE(c128, "ldr h2, [x4, w5, sxtw #1]", "", "")
ARM_CASE(c129, "ldr s2, [x4, #32]", "", "")
ARM_CASE(c130, "ldur s2, [x4, #-7]", "", "")
ARM_CASE(c131, "ldr s2, [x4, #16]!", "", "")
ARM_CASE(c132, "ldr s2, [x4], #-16", "", "")
ARM_CASE(c133, "ldr s2, [x4, x5, lsl #2]", "", "")
ARM_CASE(c134, "ldr s2, [x4, w5, sxtw #2]", "", "")
ARM_CASE(c135, "ldr d2, [x4, #32]", "", "")
ARM_CASE(c136, "ldur d2, [x4, #-7]", "", "")
ARM_CASE(c137, "ldr d2, [x4, #16]!", "", "")
ARM_CASE(c138, "ldr d2, [x4], #-16", "", "")
ARM_CASE(c139, "ldr d2, [x4, x5, lsl #3]", "", "")
ARM_CASE(c140, "ldr d2, [x4, w5, sxtw #3]", "", "")
ARM_CASE(c141, "ldr q2, [x4, #32]", "", "")
ARM_CASE(c142, "ldur q2, [x4, #-7]", "", "")
ARM_CASE(c143, "ldr q2, [x4, #16]!", "", "")
ARM_CASE(c144, "ldr q2, [x4], #-16", "", "")
ARM_CASE(c145, "ldr q2, [x4, x5, lsl #4]", "", "")
ARM_CASE(c146, "ldr q2, [x4, w5, sxtw #4]", "", "")
ARM_CASE(c147, "str b2, [x4, #32]", "", "")
ARM_CASE(c148, "stur b2, [x4, #-7]", "", "")
ARM_CASE(c149, "str b2, [x4, #16]!", "", "")
ARM_CASE(c150, "str b2, [x4], #-16", "", "")
ARM_CASE(c151, "str b2, [x4, x5, lsl #0]", "", "")
ARM_CASE(c152, "str b2, [x4, w5, sxtw #0]", "", "")
ARM_CASE(c153, "str h2, [x4, #32]", "", "")
ARM_CASE(c154, "stur h2, [x4, #-7]", "", "")
ARM_CASE(c155, "str h2, [x4, #16]!", "", "")
ARM_CASE(c156, "str h2, [x4], #-16", "", "")
ARM_CASE(c157, "str h2, [x4, x5, lsl #1]", "", "")
ARM_CASE(c158, "str h2, [x4, w5, sxtw #1]", "", "")
ARM_CASE(c159, "str s2, [x4, #32]", "", "")
ARM_CASE(c160, "stur s2, [x4, #-7]", "", "")
ARM_CASE(c161, "str s2, [x4, #16]!", "", "")
ARM_CASE(c162, "str s2, [x4], #-16", "", "")
ARM_CASE(c163, "str s2, [x4, x5, lsl #2]", "", "")
ARM_CASE(c164, "str s2, [x4, w5, sxtw #2]", "", "")
ARM_CASE(c165, "str d2, [x4, #32]", "", "")
ARM_CASE(c166, "stur d2, [x4, #-7]", "", "")
ARM_CASE(c167, "str d2, [x4, #16]!", "", "")
ARM_CASE(c168, "str d2, [x4], #-16", "", "")
ARM_CASE(c169, "str d2, [x4, x5, lsl #3]", "", "")
ARM_CASE(c170, "str d2, [x4, w5, sxtw #3]", "", "")
ARM_CASE(c171, "str q2, [x4, #32]", "", "")
ARM_CASE(c172, "stur q2, [x4, #-7]", "", "")
ARM_CASE(c173, "str q2, [x4, #16]!", "", "")
ARM_CASE(c174, "str q2, [x4], #-16", "", "")
ARM_CASE(c175, "str q2, [x4, x5, lsl #4]", "", "")
ARM_CASE(c176, "str q2, [x4, w5, sxtw #4]", "", "")
ARM_CASE(c177, "ldp w2, w3, [x4, #-32]", "", "")
ARM_CASE(c178, "ldp w2, w3, [x4, #32]!", "", "")
ARM_CASE(c179, "ldp w2, w3, [x4], #-32", "", "")
ARM_CASE(c180, "ldp x2, x3, [x4, #-32]", "", "")
ARM_CASE(c181, "ldp x2, x3, [x4, #32]!", "", "")
ARM_CASE(c182, "ldp x2, x3, [x4], #-32", "", "")
ARM_CASE(c183, "ldp s2, s3, [x4, #-32]", "", "")
ARM_CASE(c184, "ldp s2, s3, [x4, #32]!", "", "")
ARM_CASE(c185, "ldp s2, s3, [x4], #-32", "", "")
ARM_CASE(c186, "ldp d2, d3, [x4, #-32]", "", "")
ARM_CASE(c187, "ldp d2, d3, [x4, #32]!", "", "")
ARM_CASE(c188, "ldp d2, d3, [x4], #-32", "", "")
ARM_CASE(c189, "ldp q2, q3, [x4, #-32]", "", "")
ARM_CASE(c190, "ldp q2, q3, [x4, #32]!", "", "")
ARM_CASE(c191, "ldp q2, q3, [x4], #-32", "", "")
ARM_CASE(c192, "stp w2, w3, [x4, #-32]", "", "")
ARM_CASE(c193, "stp w2, w3, [x4, #32]!", "", "")
ARM_CASE(c194, "stp w2, w3, [x4], #-32", "", "")
ARM_CASE(c195, "stp x2, x3, [x4, #-32]", "", "")
ARM_CASE(c196, "stp x2, x3, [x4, #32]!", "", "")
ARM_CASE(c197, "stp x2, x3, [x4], #-32", "", "")
ARM_CASE(c198, "stp s2, s3, [x4, #-32]", "", "")
ARM_CASE(c199, "stp s2, s3, [x4, #32]!", "", "")
ARM_CASE(c200, "stp s2, s3, [x4], #-32", "", "")
ARM_CASE(c201, "stp d2, d3, [x4, #-32]", "", "")
ARM_CASE(c202, "stp d2, d3, [x4, #32]!", "", "")
ARM_CASE(c203, "stp d2, d3, [x4], #-32", "", "")
ARM_CASE(c204, "stp q2, q3, [x4, #-32]", "", "")
ARM_CASE(c205, "stp q2, q3, [x4, #32]!", "", "")
ARM_CASE(c206, "stp q2, q3, [x4], #-32", "", "")
ARM_CASE(c207, "ldnp w2, w3, [x4, #-32]", "", "")
ARM_CASE(c208, "ldnp x2, x3, [x4, #-32]", "", "")
ARM_CASE(c209, "ldnp s2, s3, [x4, #-32]", "", "")
ARM_CASE(c210, "ldnp d2, d3, [x4, #-32]", "", "")
ARM_CASE(c211, "ldnp q2, q3, [x4, #-32]", "", "")
ARM_CASE(c212, "stnp w2, w3, [x4, #-32]", "", "")
ARM_CASE(c213, "stnp x2, x3, [x4, #-32]", "", "")
ARM_CASE(c214, "stnp s2, s3, [x4, #-32]", "", "")
ARM_CASE(c215, "stnp d2, d3, [x4, #-32]", "", "")
ARM_CASE(c216, "stnp q2, q3, [x4, #-32]", "", "")
ARM_CASE(c217, "ldpsw x2, x3, [x4, #-8]", "", "")
ARM_CASE(c218, "ldpsw x2, x3, [x4, #8]!", "", "")
ARM_CASE(c219, "ldpsw x2, x3, [x4], #-8", "", "")
ARM_CASE(c220, "ldr xzr, [x4]", "", "")
ARM_CASE(c221, "ldrsb wzr, [x4]", "", "")
ARM_CASE(c222, "str xzr, [x4]", "", "")
ARM_CASE(c223, "strb wzr, [x4]", "", "")
ARM_CASE(c224, "ldp xzr, x3, [x4]", "", "")
ARM_CASE(c225, "stp xzr, x3, [x4]", "", "")
ARM_CASE(c226, "ldr q31, [x4]", "", "")
ARM_CASE(c227, "ldr x2, [sp, #16]", "mov x7, sp\nmov sp, x4\n", "mov x4, sp\nmov sp, x7\n")
ARM_CASE(c228, "str x2, [sp], #16", "mov x7, sp\nmov sp, x4\n", "mov x4, sp\nmov sp, x7\n")
ARM_CASE(c229, "ldp x2, x3, [sp, #-16]!", "mov x7, sp\nmov sp, x4\n", "mov x4, sp\nmov sp, x7\n")
ARM_CASE(c230, "stp q2, q3, [sp], #32", "mov x7, sp\nmov sp, x4\n", "mov x4, sp\nmov sp, x7\n")
ARM_CASE(c231, "ldr q2, [sp, w5, sxtw #4]", "mov x7, sp\nmov sp, x4\n", "mov x4, sp\nmov sp, x7\n")
ARM_CASE(c232, "ldr q2, [sp, #16]!", "mov x7, sp\nmov sp, x4\n", "mov x4, sp\nmov sp, x7\n")
ARM_CASE(c233, "ldr w2, 2f\nb 3f\n.p2align 4\n2: .quad 0x80fedcba87654321, 0xaabbccdd00112233\n3:", "", "")
ARM_CASE(c234, "ldr x2, 2f\nb 3f\n.p2align 4\n2: .quad 0x80fedcba87654321, 0xaabbccdd00112233\n3:", "", "")
ARM_CASE(c235, "ldrsw x2, 2f\nb 3f\n.p2align 4\n2: .quad 0x80fedcba87654321, 0xaabbccdd00112233\n3:", "", "")
ARM_CASE(c236, "ldr s2, 2f\nb 3f\n.p2align 4\n2: .quad 0x80fedcba87654321, 0xaabbccdd00112233\n3:", "", "")
ARM_CASE(c237, "ldr d2, 2f\nb 3f\n.p2align 4\n2: .quad 0x80fedcba87654321, 0xaabbccdd00112233\n3:", "", "")
ARM_CASE(c238, "ldr q2, 2f\nb 3f\n.p2align 4\n2: .quad 0x80fedcba87654321, 0xaabbccdd00112233\n3:", "", "")
#undef ARM_CASE
struct probe { const char* instruction; probe_type execute; bool sp, literal; std::uint64_t index; };
const probe probes[]{
	{"ldrb w2, [x4, #16]", c0, false, false, 0x10ull},
	{"ldurb w2, [x4, #-7]", c1, false, false, 0x10ull},
	{"ldrb w2, [x4, #16]!", c2, false, false, 0x10ull},
	{"ldrb w2, [x4], #-16", c3, false, false, 0x10ull},
	{"ldrb w2, [x4, x5]", c4, false, false, 0x10ull},
	{"ldrb w2, [x4, w5, uxtw]", c5, false, false, 0x10ull},
	{"ldrb w2, [x4, w5, sxtw]", c6, false, false, 0xfffffff0ull},
	{"ldrb w2, [x4, x5, sxtx]", c7, false, false, 0xfffffffffffffff0ull},
	{"ldrb w2, [x4, x5, lsl #0]", c8, false, false, 0x2ull},
	{"ldrh w2, [x4, #16]", c9, false, false, 0x10ull},
	{"ldurh w2, [x4, #-7]", c10, false, false, 0x10ull},
	{"ldrh w2, [x4, #16]!", c11, false, false, 0x10ull},
	{"ldrh w2, [x4], #-16", c12, false, false, 0x10ull},
	{"ldrh w2, [x4, x5]", c13, false, false, 0x10ull},
	{"ldrh w2, [x4, w5, uxtw]", c14, false, false, 0x10ull},
	{"ldrh w2, [x4, w5, sxtw]", c15, false, false, 0xfffffff0ull},
	{"ldrh w2, [x4, x5, sxtx]", c16, false, false, 0xfffffffffffffff0ull},
	{"ldrh w2, [x4, x5, lsl #1]", c17, false, false, 0x2ull},
	{"ldr w2, [x4, #16]", c18, false, false, 0x10ull},
	{"ldur w2, [x4, #-7]", c19, false, false, 0x10ull},
	{"ldr w2, [x4, #16]!", c20, false, false, 0x10ull},
	{"ldr w2, [x4], #-16", c21, false, false, 0x10ull},
	{"ldr w2, [x4, x5]", c22, false, false, 0x10ull},
	{"ldr w2, [x4, w5, uxtw]", c23, false, false, 0x10ull},
	{"ldr w2, [x4, w5, sxtw]", c24, false, false, 0xfffffff0ull},
	{"ldr w2, [x4, x5, sxtx]", c25, false, false, 0xfffffffffffffff0ull},
	{"ldr w2, [x4, x5, lsl #2]", c26, false, false, 0x2ull},
	{"ldr x2, [x4, #16]", c27, false, false, 0x10ull},
	{"ldur x2, [x4, #-7]", c28, false, false, 0x10ull},
	{"ldr x2, [x4, #16]!", c29, false, false, 0x10ull},
	{"ldr x2, [x4], #-16", c30, false, false, 0x10ull},
	{"ldr x2, [x4, x5]", c31, false, false, 0x10ull},
	{"ldr x2, [x4, w5, uxtw]", c32, false, false, 0x10ull},
	{"ldr x2, [x4, w5, sxtw]", c33, false, false, 0xfffffff0ull},
	{"ldr x2, [x4, x5, sxtx]", c34, false, false, 0xfffffffffffffff0ull},
	{"ldr x2, [x4, x5, lsl #3]", c35, false, false, 0x2ull},
	{"ldrsb w2, [x4, #16]", c36, false, false, 0x10ull},
	{"ldursb w2, [x4, #-7]", c37, false, false, 0x10ull},
	{"ldrsb w2, [x4, #16]!", c38, false, false, 0x10ull},
	{"ldrsb w2, [x4], #-16", c39, false, false, 0x10ull},
	{"ldrsb w2, [x4, x5]", c40, false, false, 0x10ull},
	{"ldrsb w2, [x4, w5, uxtw]", c41, false, false, 0x10ull},
	{"ldrsb w2, [x4, w5, sxtw]", c42, false, false, 0xfffffff0ull},
	{"ldrsb w2, [x4, x5, sxtx]", c43, false, false, 0xfffffffffffffff0ull},
	{"ldrsb w2, [x4, x5, lsl #0]", c44, false, false, 0x2ull},
	{"ldrsb x2, [x4, #16]", c45, false, false, 0x10ull},
	{"ldursb x2, [x4, #-7]", c46, false, false, 0x10ull},
	{"ldrsb x2, [x4, #16]!", c47, false, false, 0x10ull},
	{"ldrsb x2, [x4], #-16", c48, false, false, 0x10ull},
	{"ldrsb x2, [x4, x5]", c49, false, false, 0x10ull},
	{"ldrsb x2, [x4, w5, uxtw]", c50, false, false, 0x10ull},
	{"ldrsb x2, [x4, w5, sxtw]", c51, false, false, 0xfffffff0ull},
	{"ldrsb x2, [x4, x5, sxtx]", c52, false, false, 0xfffffffffffffff0ull},
	{"ldrsb x2, [x4, x5, lsl #0]", c53, false, false, 0x2ull},
	{"ldrsh w2, [x4, #16]", c54, false, false, 0x10ull},
	{"ldursh w2, [x4, #-7]", c55, false, false, 0x10ull},
	{"ldrsh w2, [x4, #16]!", c56, false, false, 0x10ull},
	{"ldrsh w2, [x4], #-16", c57, false, false, 0x10ull},
	{"ldrsh w2, [x4, x5]", c58, false, false, 0x10ull},
	{"ldrsh w2, [x4, w5, uxtw]", c59, false, false, 0x10ull},
	{"ldrsh w2, [x4, w5, sxtw]", c60, false, false, 0xfffffff0ull},
	{"ldrsh w2, [x4, x5, sxtx]", c61, false, false, 0xfffffffffffffff0ull},
	{"ldrsh w2, [x4, x5, lsl #1]", c62, false, false, 0x2ull},
	{"ldrsh x2, [x4, #16]", c63, false, false, 0x10ull},
	{"ldursh x2, [x4, #-7]", c64, false, false, 0x10ull},
	{"ldrsh x2, [x4, #16]!", c65, false, false, 0x10ull},
	{"ldrsh x2, [x4], #-16", c66, false, false, 0x10ull},
	{"ldrsh x2, [x4, x5]", c67, false, false, 0x10ull},
	{"ldrsh x2, [x4, w5, uxtw]", c68, false, false, 0x10ull},
	{"ldrsh x2, [x4, w5, sxtw]", c69, false, false, 0xfffffff0ull},
	{"ldrsh x2, [x4, x5, sxtx]", c70, false, false, 0xfffffffffffffff0ull},
	{"ldrsh x2, [x4, x5, lsl #1]", c71, false, false, 0x2ull},
	{"ldrsw x2, [x4, #16]", c72, false, false, 0x10ull},
	{"ldursw x2, [x4, #-7]", c73, false, false, 0x10ull},
	{"ldrsw x2, [x4, #16]!", c74, false, false, 0x10ull},
	{"ldrsw x2, [x4], #-16", c75, false, false, 0x10ull},
	{"ldrsw x2, [x4, x5]", c76, false, false, 0x10ull},
	{"ldrsw x2, [x4, w5, uxtw]", c77, false, false, 0x10ull},
	{"ldrsw x2, [x4, w5, sxtw]", c78, false, false, 0xfffffff0ull},
	{"ldrsw x2, [x4, x5, sxtx]", c79, false, false, 0xfffffffffffffff0ull},
	{"ldrsw x2, [x4, x5, lsl #2]", c80, false, false, 0x2ull},
	{"strb w2, [x4, #16]", c81, false, false, 0x10ull},
	{"sturb w2, [x4, #-7]", c82, false, false, 0x10ull},
	{"strb w2, [x4, #16]!", c83, false, false, 0x10ull},
	{"strb w2, [x4], #-16", c84, false, false, 0x10ull},
	{"strb w2, [x4, x5]", c85, false, false, 0x10ull},
	{"strb w2, [x4, w5, uxtw]", c86, false, false, 0x10ull},
	{"strb w2, [x4, w5, sxtw]", c87, false, false, 0xfffffff0ull},
	{"strb w2, [x4, x5, sxtx]", c88, false, false, 0xfffffffffffffff0ull},
	{"strb w2, [x4, x5, lsl #0]", c89, false, false, 0x2ull},
	{"strh w2, [x4, #16]", c90, false, false, 0x10ull},
	{"sturh w2, [x4, #-7]", c91, false, false, 0x10ull},
	{"strh w2, [x4, #16]!", c92, false, false, 0x10ull},
	{"strh w2, [x4], #-16", c93, false, false, 0x10ull},
	{"strh w2, [x4, x5]", c94, false, false, 0x10ull},
	{"strh w2, [x4, w5, uxtw]", c95, false, false, 0x10ull},
	{"strh w2, [x4, w5, sxtw]", c96, false, false, 0xfffffff0ull},
	{"strh w2, [x4, x5, sxtx]", c97, false, false, 0xfffffffffffffff0ull},
	{"strh w2, [x4, x5, lsl #1]", c98, false, false, 0x2ull},
	{"str w2, [x4, #16]", c99, false, false, 0x10ull},
	{"stur w2, [x4, #-7]", c100, false, false, 0x10ull},
	{"str w2, [x4, #16]!", c101, false, false, 0x10ull},
	{"str w2, [x4], #-16", c102, false, false, 0x10ull},
	{"str w2, [x4, x5]", c103, false, false, 0x10ull},
	{"str w2, [x4, w5, uxtw]", c104, false, false, 0x10ull},
	{"str w2, [x4, w5, sxtw]", c105, false, false, 0xfffffff0ull},
	{"str w2, [x4, x5, sxtx]", c106, false, false, 0xfffffffffffffff0ull},
	{"str w2, [x4, x5, lsl #2]", c107, false, false, 0x2ull},
	{"str x2, [x4, #16]", c108, false, false, 0x10ull},
	{"stur x2, [x4, #-7]", c109, false, false, 0x10ull},
	{"str x2, [x4, #16]!", c110, false, false, 0x10ull},
	{"str x2, [x4], #-16", c111, false, false, 0x10ull},
	{"str x2, [x4, x5]", c112, false, false, 0x10ull},
	{"str x2, [x4, w5, uxtw]", c113, false, false, 0x10ull},
	{"str x2, [x4, w5, sxtw]", c114, false, false, 0xfffffff0ull},
	{"str x2, [x4, x5, sxtx]", c115, false, false, 0xfffffffffffffff0ull},
	{"str x2, [x4, x5, lsl #3]", c116, false, false, 0x2ull},
	{"ldr b2, [x4, #32]", c117, false, false, 0x2ull},
	{"ldur b2, [x4, #-7]", c118, false, false, 0x2ull},
	{"ldr b2, [x4, #16]!", c119, false, false, 0x2ull},
	{"ldr b2, [x4], #-16", c120, false, false, 0x2ull},
	{"ldr b2, [x4, x5, lsl #0]", c121, false, false, 0x2ull},
	{"ldr b2, [x4, w5, sxtw #0]", c122, false, false, 0xfffffffeull},
	{"ldr h2, [x4, #32]", c123, false, false, 0x2ull},
	{"ldur h2, [x4, #-7]", c124, false, false, 0x2ull},
	{"ldr h2, [x4, #16]!", c125, false, false, 0x2ull},
	{"ldr h2, [x4], #-16", c126, false, false, 0x2ull},
	{"ldr h2, [x4, x5, lsl #1]", c127, false, false, 0x2ull},
	{"ldr h2, [x4, w5, sxtw #1]", c128, false, false, 0xfffffffeull},
	{"ldr s2, [x4, #32]", c129, false, false, 0x2ull},
	{"ldur s2, [x4, #-7]", c130, false, false, 0x2ull},
	{"ldr s2, [x4, #16]!", c131, false, false, 0x2ull},
	{"ldr s2, [x4], #-16", c132, false, false, 0x2ull},
	{"ldr s2, [x4, x5, lsl #2]", c133, false, false, 0x2ull},
	{"ldr s2, [x4, w5, sxtw #2]", c134, false, false, 0xfffffffeull},
	{"ldr d2, [x4, #32]", c135, false, false, 0x2ull},
	{"ldur d2, [x4, #-7]", c136, false, false, 0x2ull},
	{"ldr d2, [x4, #16]!", c137, false, false, 0x2ull},
	{"ldr d2, [x4], #-16", c138, false, false, 0x2ull},
	{"ldr d2, [x4, x5, lsl #3]", c139, false, false, 0x2ull},
	{"ldr d2, [x4, w5, sxtw #3]", c140, false, false, 0xfffffffeull},
	{"ldr q2, [x4, #32]", c141, false, false, 0x2ull},
	{"ldur q2, [x4, #-7]", c142, false, false, 0x2ull},
	{"ldr q2, [x4, #16]!", c143, false, false, 0x2ull},
	{"ldr q2, [x4], #-16", c144, false, false, 0x2ull},
	{"ldr q2, [x4, x5, lsl #4]", c145, false, false, 0x2ull},
	{"ldr q2, [x4, w5, sxtw #4]", c146, false, false, 0xfffffffeull},
	{"str b2, [x4, #32]", c147, false, false, 0x2ull},
	{"stur b2, [x4, #-7]", c148, false, false, 0x2ull},
	{"str b2, [x4, #16]!", c149, false, false, 0x2ull},
	{"str b2, [x4], #-16", c150, false, false, 0x2ull},
	{"str b2, [x4, x5, lsl #0]", c151, false, false, 0x2ull},
	{"str b2, [x4, w5, sxtw #0]", c152, false, false, 0xfffffffeull},
	{"str h2, [x4, #32]", c153, false, false, 0x2ull},
	{"stur h2, [x4, #-7]", c154, false, false, 0x2ull},
	{"str h2, [x4, #16]!", c155, false, false, 0x2ull},
	{"str h2, [x4], #-16", c156, false, false, 0x2ull},
	{"str h2, [x4, x5, lsl #1]", c157, false, false, 0x2ull},
	{"str h2, [x4, w5, sxtw #1]", c158, false, false, 0xfffffffeull},
	{"str s2, [x4, #32]", c159, false, false, 0x2ull},
	{"stur s2, [x4, #-7]", c160, false, false, 0x2ull},
	{"str s2, [x4, #16]!", c161, false, false, 0x2ull},
	{"str s2, [x4], #-16", c162, false, false, 0x2ull},
	{"str s2, [x4, x5, lsl #2]", c163, false, false, 0x2ull},
	{"str s2, [x4, w5, sxtw #2]", c164, false, false, 0xfffffffeull},
	{"str d2, [x4, #32]", c165, false, false, 0x2ull},
	{"stur d2, [x4, #-7]", c166, false, false, 0x2ull},
	{"str d2, [x4, #16]!", c167, false, false, 0x2ull},
	{"str d2, [x4], #-16", c168, false, false, 0x2ull},
	{"str d2, [x4, x5, lsl #3]", c169, false, false, 0x2ull},
	{"str d2, [x4, w5, sxtw #3]", c170, false, false, 0xfffffffeull},
	{"str q2, [x4, #32]", c171, false, false, 0x2ull},
	{"stur q2, [x4, #-7]", c172, false, false, 0x2ull},
	{"str q2, [x4, #16]!", c173, false, false, 0x2ull},
	{"str q2, [x4], #-16", c174, false, false, 0x2ull},
	{"str q2, [x4, x5, lsl #4]", c175, false, false, 0x2ull},
	{"str q2, [x4, w5, sxtw #4]", c176, false, false, 0xfffffffeull},
	{"ldp w2, w3, [x4, #-32]", c177, false, false, 0x10ull},
	{"ldp w2, w3, [x4, #32]!", c178, false, false, 0x10ull},
	{"ldp w2, w3, [x4], #-32", c179, false, false, 0x10ull},
	{"ldp x2, x3, [x4, #-32]", c180, false, false, 0x10ull},
	{"ldp x2, x3, [x4, #32]!", c181, false, false, 0x10ull},
	{"ldp x2, x3, [x4], #-32", c182, false, false, 0x10ull},
	{"ldp s2, s3, [x4, #-32]", c183, false, false, 0x10ull},
	{"ldp s2, s3, [x4, #32]!", c184, false, false, 0x10ull},
	{"ldp s2, s3, [x4], #-32", c185, false, false, 0x10ull},
	{"ldp d2, d3, [x4, #-32]", c186, false, false, 0x10ull},
	{"ldp d2, d3, [x4, #32]!", c187, false, false, 0x10ull},
	{"ldp d2, d3, [x4], #-32", c188, false, false, 0x10ull},
	{"ldp q2, q3, [x4, #-32]", c189, false, false, 0x10ull},
	{"ldp q2, q3, [x4, #32]!", c190, false, false, 0x10ull},
	{"ldp q2, q3, [x4], #-32", c191, false, false, 0x10ull},
	{"stp w2, w3, [x4, #-32]", c192, false, false, 0x10ull},
	{"stp w2, w3, [x4, #32]!", c193, false, false, 0x10ull},
	{"stp w2, w3, [x4], #-32", c194, false, false, 0x10ull},
	{"stp x2, x3, [x4, #-32]", c195, false, false, 0x10ull},
	{"stp x2, x3, [x4, #32]!", c196, false, false, 0x10ull},
	{"stp x2, x3, [x4], #-32", c197, false, false, 0x10ull},
	{"stp s2, s3, [x4, #-32]", c198, false, false, 0x10ull},
	{"stp s2, s3, [x4, #32]!", c199, false, false, 0x10ull},
	{"stp s2, s3, [x4], #-32", c200, false, false, 0x10ull},
	{"stp d2, d3, [x4, #-32]", c201, false, false, 0x10ull},
	{"stp d2, d3, [x4, #32]!", c202, false, false, 0x10ull},
	{"stp d2, d3, [x4], #-32", c203, false, false, 0x10ull},
	{"stp q2, q3, [x4, #-32]", c204, false, false, 0x10ull},
	{"stp q2, q3, [x4, #32]!", c205, false, false, 0x10ull},
	{"stp q2, q3, [x4], #-32", c206, false, false, 0x10ull},
	{"ldnp w2, w3, [x4, #-32]", c207, false, false, 0x10ull},
	{"ldnp x2, x3, [x4, #-32]", c208, false, false, 0x10ull},
	{"ldnp s2, s3, [x4, #-32]", c209, false, false, 0x10ull},
	{"ldnp d2, d3, [x4, #-32]", c210, false, false, 0x10ull},
	{"ldnp q2, q3, [x4, #-32]", c211, false, false, 0x10ull},
	{"stnp w2, w3, [x4, #-32]", c212, false, false, 0x10ull},
	{"stnp x2, x3, [x4, #-32]", c213, false, false, 0x10ull},
	{"stnp s2, s3, [x4, #-32]", c214, false, false, 0x10ull},
	{"stnp d2, d3, [x4, #-32]", c215, false, false, 0x10ull},
	{"stnp q2, q3, [x4, #-32]", c216, false, false, 0x10ull},
	{"ldpsw x2, x3, [x4, #-8]", c217, false, false, 0x10ull},
	{"ldpsw x2, x3, [x4, #8]!", c218, false, false, 0x10ull},
	{"ldpsw x2, x3, [x4], #-8", c219, false, false, 0x10ull},
	{"ldr xzr, [x4]", c220, false, false, 0x10ull},
	{"ldrsb wzr, [x4]", c221, false, false, 0x10ull},
	{"str xzr, [x4]", c222, false, false, 0x10ull},
	{"strb wzr, [x4]", c223, false, false, 0x10ull},
	{"ldp xzr, x3, [x4]", c224, false, false, 0x10ull},
	{"stp xzr, x3, [x4]", c225, false, false, 0x10ull},
	{"ldr q31, [x4]", c226, false, false, 0x10ull},
	{"ldr x2, [sp, #16]", c227, true, false, 0x10ull},
	{"str x2, [sp], #16", c228, true, false, 0x10ull},
	{"ldp x2, x3, [sp, #-16]!", c229, true, false, 0x10ull},
	{"stp q2, q3, [sp], #32", c230, true, false, 0x10ull},
	{"ldr q2, [sp, w5, sxtw #4]", c231, true, false, 0xfffffffeull},
	{"ldr q2, [sp, #16]!", c232, true, false, 0x10ull},
	{"ldr w2, 2f", c233, false, true, 0x10ull},
	{"ldr x2, 2f", c234, false, true, 0x10ull},
	{"ldrsw x2, 2f", c235, false, true, 0x10ull},
	{"ldr s2, 2f", c236, false, true, 0x10ull},
	{"ldr d2, 2f", c237, false, true, 0x10ull},
	{"ldr q2, 2f", c238, false, true, 0x10ull},
};
std::uint64_t random(std::uint64_t& value)
{
	value ^= value << 13; value ^= value >> 7; value ^= value << 17;
	return value;
}
void test_decode()
{
	std::uint64_t rng = 0x4d6574616c344152ull;
	unsigned checked = 0;
	for (const auto& p : probes)
		for (unsigned iteration = 0; iteration < 32; ++iteration)
		{
			alignas(32) std::array<std::byte, 1024> storage;
			for (auto& b : storage) b = std::byte(random(rng));
			const auto original = storage;
			arm64_memory_state before;
			for (auto& x : before.x) x = random(rng);
			for (auto& v : before.v) for (auto& b : v) b = std::byte(random(rng));
			before.x[4] = reinterpret_cast<std::uint64_t>(storage.data() + 512);
			before.x[5] = p.index;
			before.sp = p.sp ? before.x[4] : 0x01234000;
			before.cpsr = 0xa0000000; before.fpcr = 0x01000000; before.fpsr = 0x08000000;
			vector_pair vectors;
			std::copy(before.v[2].begin(), before.v[2].end(), vectors.begin());
			std::copy(before.v[3].begin(), before.v[3].end(), vectors.begin() + 16);
			std::copy(before.v[31].begin(), before.v[31].end(), vectors.begin() + 32);
			const auto actual = p.execute(before.x[4], before.x[5], before.x[2], before.x[3], vectors);
			before.pc = actual.pc;
			std::uint32_t instruction;
			std::memcpy(&instruction, reinterpret_cast<const void*>(actual.pc), 4);
			const auto request = arm64_memory_request::decode(instruction, before);
			if (!request) throw std::runtime_error(std::string("Actual ARM instruction refused: ") + p.instruction);
			const auto begin = p.literal ? actual.pc : reinterpret_cast<std::uint64_t>(storage.data());
			const auto extent = p.literal ? 64u : storage.size();
			require(request->address() >= begin && request->size() <= extent && request->address() - begin <= extent - request->size(), "ARM effective address escaped the independently owned hardware fixture");
			arm64_memory_state result;
			if (request->writing())
			{
				auto expected = original;
				const auto payload = request->store_bytes();
				std::copy_n(payload.begin(), request->size(), expected.begin() + (request->address() - begin));
				require(storage == expected, "ARM store payload/address/extent differs from actual hardware");
				result = request->complete({});
			}
			else
			{
				require(storage == original, "Actual ARM load unexpectedly changed fixture bytes");
				result = request->complete({reinterpret_cast<const std::byte*>(request->address()), request->size()});
			}
			auto expected = before;
			expected.x[2] = actual.first; expected.x[3] = actual.second;
			if (p.sp) expected.sp = actual.base; else expected.x[4] = actual.base;
			std::copy_n(actual.vectors.begin(), 16, expected.v[2].begin());
			std::copy_n(actual.vectors.begin() + 16, 16, expected.v[3].begin());
			std::copy_n(actual.vectors.begin() + 32, 16, expected.v[31].begin());
			expected.pc += 4;
			if (result != expected) throw std::runtime_error(std::string("ARM register/writeback/zero/sign extension differs from hardware: ") + p.instruction);
			for (unsigned i = 0; i < request->size(); ++i) require(request->matches_fault(request->address() + i, request->writing()), "ARM fault byte lost its whole instruction extent");
			require(!request->matches_fault(request->address() - 1, request->writing()) && !request->matches_fault(request->address() + request->size(), request->writing()) && !request->matches_fault(request->address(), !request->writing()), "ARM fault accepted an unrelated byte or direction");
			++checked;
		}
	require(checked == std::size(probes) * 32, "ARM hardware matrix was truncated");
	std::cout << "ARM hardware matrix:" << std::size(probes) << " instruction forms," << checked << " randomized executions;scalar/SIMD/paired/literal/index/writeback/SP/XZR;no fault or guest CPU\n";
}
void test_refusal_and_ownership()
{
	arm64_memory_state before;
	before.pc = 0x1000; before.x[4] = 0x10000; before.sp = 0x20000;
	before.x[2] = 0x123456789abcdef0;
	for (auto instruction : {0xc85f7c82u, 0xc8037c82u, 0xc8dffc82u, 0xc89ffc82u, 0xf8400882u, 0xf8000882u, 0x4c407082u, 0x4c007082u, 0xf9800080u, 0xd503201fu, 0xa9400882u, 0xf8408484u, 0xb8c00082u, 0x3dc00082u | (1u << 30)})
		require(!arm64_memory_request::decode(instruction, before), "ARM unsupported/reserved/constrained instruction did not refuse");
	const auto original = before;
	const auto request = arm64_memory_request::decode(0xf9000082, before); // STR X2,[X4]
	require(bool(request), "Owned ARM store request refused");
	before.x[2] = 0; before.x[4] = 0;
	require(request->address() == original.x[4] && request->store_bytes()[0] == std::byte{0xf0} && request->complete({}).x[2] == original.x[2], "ARM request borrowed mutable caller registers");
	bool refused = false;
	try { (void)request->complete(std::array<std::byte, 1>{}); } catch (const std::invalid_argument&) { refused = true; }
	require(refused, "ARM reply accepted a wrong exact extent");
	before = original; before.pc += 1; require(!arm64_memory_request::decode(0xf9000082, before), "ARM unaligned PC accepted");
	before = original; before.pc = ~std::uint64_t{3}; require(!arm64_memory_request::decode(0xf9000082, before), "ARM PC overflow accepted");
	before = original; before.x[4] = ~std::uint64_t{3}; require(!arm64_memory_request::decode(0xf9000082, before), "ARM access extent overflow accepted");
	before = original; before.x[4] = 4; require(!arm64_memory_request::decode(0xf85f8082, before), "ARM negative effective-address wrap accepted");
	before = original; before.sp += 1; require(!arm64_memory_request::decode(0xf94003e2, before), "ARM misaligned SP accepted");
	std::cout << "ARM refusal:atomics/exclusives/unprivileged/structures/prefetch/reserved/writeback overlap;whole extent/PC/SP/owned registers\n";
}
// Controlled real Darwin faults on an independent shared test mapping. The
// signal handler only copies fixed state and waits;the owner decodes/completes
// the instruction. No guest VM/GSRender/App hook is installed by this harness.
struct fault_exchange
{
	static_assert(std::atomic<unsigned>::is_always_lock_free);
	std::atomic<unsigned> phase{0}; // 1=captured,2=reply,3=completed,4=cancel
	arm64_memory_state before, after;
	std::uint64_t address = 0;
	std::uint32_t syndrome = 0;
	hardware_result actual{};
};
thread_local fault_exchange* current_fault = nullptr;
void controlled_fault(int, siginfo_t* info, void* opaque)
{
	auto* exchange = current_fault;
	auto* context = static_cast<ucontext_t*>(opaque);
	if (!exchange || !info || !context) ::_exit(80);
	const auto before = capture_arm64_context(*context);
	if (!before) ::_exit(81);
	exchange->before = *before;
	exchange->address = reinterpret_cast<std::uint64_t>(info->si_addr);
	exchange->syndrome = context->uc_mcontext->__es.__esr;
	unsigned waiting = 0;
	if (!exchange->phase.compare_exchange_strong(waiting, 1, std::memory_order_release, std::memory_order_relaxed)) ::_exit(85);
	while (exchange->phase.load(std::memory_order_acquire) == 1) asm volatile("yield");
	if (exchange->phase.load(std::memory_order_acquire) != 2 || !publish_arm64_context(*context, exchange->before, exchange->after)) ::_exit(82);
}
void test_actual_fault_context()
{
	const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
	require(page >= 4096 && page <= 65536, "Controlled ARM fault test has unsupported host pages");
	char path[] = "/tmp/rpcs3-metal4-arm-context-XXXXXX";
	const auto file = ::mkstemp(path);
	require(file >= 0, "Could not create controlled ARM fault test storage");
	(void)::unlink(path);
	if (::ftruncate(file, static_cast<off_t>(2 * page)) != 0) { ::close(file); throw std::runtime_error("Could not size controlled ARM fault test storage"); }
	auto* canonical = static_cast<std::byte*>(::mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0));
	auto* trapped = static_cast<std::byte*>(::mmap(nullptr, 2 * page, PROT_NONE, MAP_SHARED, file, 0));
	::close(file);
	if (canonical == MAP_FAILED || trapped == MAP_FAILED)
	{
		if (canonical != MAP_FAILED) (void)::munmap(canonical, 2 * page);
		if (trapped != MAP_FAILED) (void)::munmap(trapped, 2 * page);
		throw std::runtime_error("Could not map controlled ARM fault aliases");
	}
	struct mappings
	{
		std::byte* canonical; std::byte* trapped; std::size_t size;
		~mappings() { (void)::munmap(canonical, size); (void)::munmap(trapped, size); }
	} cleanup{canonical, trapped, 2 * page};
	struct sigaction action{}, previous_bus{}, previous_segv{};
	action.sa_sigaction = controlled_fault; action.sa_flags = SA_SIGINFO | SA_ONSTACK;
	(void)sigemptyset(&action.sa_mask);
	require(::sigaction(SIGBUS, &action, &previous_bus) == 0, "Could not install controlled ARM bus-fault harness");
	if (::sigaction(SIGSEGV, &action, &previous_segv) != 0)
	{
		(void)::sigaction(SIGBUS, &previous_bus, nullptr);
		throw std::runtime_error("Could not install controlled ARM segv-fault harness");
	}
	struct signals
	{
		struct sigaction bus, segv;
		~signals() { (void)::sigaction(SIGBUS, &bus, nullptr); (void)::sigaction(SIGSEGV, &segv, nullptr); }
	} restore{previous_bus, previous_segv};
	unsigned completed = 0;
	for (const auto& p : probes)
	{
		if (p.literal) continue; // Literal fixtures reference executable text.
		// Cover every ordinary addressing/extension/width, including actual SP
		// and XZR faults. All pages stay PROT_NONE until the mappings retire.
		for (std::size_t i = 0; i < 2 * page; ++i) canonical[i] = std::byte((i * 131 + completed * 17) & 255);
		vector_pair vectors;
		for (unsigned i = 0; i < vectors.size(); ++i) vectors[i] = std::byte(i * 53 + completed);
		const std::uint64_t first = 0x8070605040302010ull ^ completed, second = 0xfedcba9876543210ull ^ completed;
		// Paired accesses start eight bytes before a host-page boundary;negative
		// offsets still remain in the independently owned two-page fixture.
		const auto offset = p.sp ? page : page - 8;
		std::vector<std::byte> original(canonical, canonical + 2 * page);
		const auto expected = p.execute(reinterpret_cast<std::uint64_t>(canonical + offset), p.index, first, second, vectors);
		std::vector<std::byte> expected_bytes(canonical, canonical + 2 * page);
		std::copy(original.begin(), original.end(), canonical);
		fault_exchange exchange;
		std::thread producer([&]
		{
			// Alternate stack is essential when the faulting instruction uses SP
			// as a protected guest address. Its lifetime includes signal return.
			std::array<std::byte, 65536> signal_stack;
			stack_t alternate{}; alternate.ss_sp = signal_stack.data(); alternate.ss_size = signal_stack.size();
			stack_t previous_stack{};
			if (::sigaltstack(&alternate, &previous_stack) != 0) ::_exit(83);
			// SA_ONSTACK is already installed process-wide below;this stack is
			// per thread. No allocation or mutex operation occurs in the handler.
			current_fault = &exchange;
			exchange.actual = p.execute(reinterpret_cast<std::uint64_t>(trapped + offset), p.index, first, second, vectors);
			current_fault = nullptr;
			// Darwin validates the minimum size even when disabling a stack.
			// Restore an active sanitizer stack exactly;for an originally disabled
			// stack restore its disabled state with the required minimum size.
			const auto original_stack = previous_stack;
			if (previous_stack.ss_flags & SS_DISABLE) previous_stack.ss_size = std::max(previous_stack.ss_size, static_cast<std::size_t>(MINSIGSTKSZ));
			if (::sigaltstack(&previous_stack, nullptr) != 0) ::_exit(84);
			stack_t restored_stack{};
			if (::sigaltstack(nullptr, &restored_stack) != 0 ||
				((original_stack.ss_flags & SS_DISABLE) ? !(restored_stack.ss_flags & SS_DISABLE) :
				(restored_stack.ss_flags != original_stack.ss_flags || restored_stack.ss_sp != original_stack.ss_sp || restored_stack.ss_size != original_stack.ss_size))) ::_exit(86);
			exchange.phase.store(3, std::memory_order_release);
		});
		try
		{
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			while (!exchange.phase.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
			require(exchange.phase.load(std::memory_order_acquire) == 1, "Controlled ARM instruction did not deliver its actual fault context");
			std::uint32_t instruction;
			std::memcpy(&instruction, reinterpret_cast<const void*>(exchange.before.pc), 4);
			const auto request = arm64_memory_request::decode(instruction, exchange.before);
			const auto exception = exchange.syndrome >> 26;
			require(request && (exception == 0x24 || exception == 0x25) && request->matches_fault(exchange.address, (exchange.syndrome & 64) != 0), "Actual Darwin data-abort address/direction disagrees with complete ARM decode");
			const auto begin = reinterpret_cast<std::uint64_t>(trapped);
			require(request->address() >= begin && request->address() - begin <= 2 * page - request->size(), "Actual ARM fault escaped its shared mapping");
			auto* bytes = canonical + (request->address() - begin);
			if (request->writing())
			{
				const auto payload = request->store_bytes();
				exchange.after = request->complete({});
				std::copy_n(payload.begin(), request->size(), bytes);
			}
			else exchange.after = request->complete({bytes, request->size()});
			exchange.phase.store(2, std::memory_order_release);
			while (exchange.phase.load(std::memory_order_acquire) != 3 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
			require(exchange.phase.load(std::memory_order_acquire) == 3, "Actual Darwin signal reply did not resume after the exact instruction");
			producer.join();
			require(exchange.actual.first == expected.first && exchange.actual.second == expected.second && exchange.actual.vectors == expected.vectors &&
				exchange.actual.pc == expected.pc && exchange.actual.base - reinterpret_cast<std::uint64_t>(trapped) == expected.base - reinterpret_cast<std::uint64_t>(canonical) &&
				std::equal(expected_bytes.begin(), expected_bytes.end(), canonical), "Actual Darwin resumed registers/SP/payload differ from nonfaulting hardware");
		}
		catch (...)
		{
			exchange.phase.store(4, std::memory_order_release);
			// A failed reply must never resume a protected hardware instruction.
			// The handler terminates this isolated headless test instead.
			producer.join();
			throw;
		}
		++completed;
	}
	require(completed == std::size(probes) - 6, "Actual Darwin fault matrix omitted instruction forms");
	mach_vm_address_t region = reinterpret_cast<mach_vm_address_t>(trapped);
	const auto requested = region;
	mach_vm_size_t region_size = 0;
	vm_region_basic_info_data_64_t info{};
	mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
	mach_port_t object = MACH_PORT_NULL;
	const auto status = ::mach_vm_region(mach_task_self(), &region, &region_size, VM_REGION_BASIC_INFO_64,
		reinterpret_cast<vm_region_info_t>(&info), &count, &object);
	if (object != MACH_PORT_NULL) (void)::mach_port_deallocate(mach_task_self(), object);
	require(status == KERN_SUCCESS && region <= requested && requested - region <= region_size &&
		2 * page <= region_size - (requested - region) && info.protection == VM_PROT_NONE, "Actual ARM signal completion opened its protected CPU pages");
	std::cout << "ARM actual Darwin faults:" << completed << " scalar/SIMD/pair/index/SP/XZR forms;owned parked host thread and exact register/PC reply;shared canonical bytes;both host pages remain NO;no VM/cache/App signal installation\n";
}

void test_context()
{
	std::remove_pointer_t<mcontext_t> machine{};
	ucontext_t context{};
	context.uc_mcontext = &machine; context.uc_mcsize = sizeof(machine);
	machine.__ss.__pc = 0x1000; machine.__ss.__sp = 0x2000;
	machine.__ss.__fp = 0xdeadbeef12345678; machine.__ss.__lr = 0xface1234abcdef80;
	machine.__ss.__cpsr = 0xa0000000; machine.__ns.__fpsr = 0x08000000; machine.__ns.__fpcr = 0x01000000;
	for (unsigned i = 0; i < 29; ++i) machine.__ss.__x[i] = i * 0x123456789ull;
	for (unsigned i = 0; i < 32; ++i) machine.__ns.__v[i] = (__uint128_t(i + 1) << 100) | (i * 0xaabbccddull);
	machine.__es.__far = 0x1122334455667788; machine.__es.__esr = 0x87654321; machine.__ss.__pad = 0x99aabbcc;
	context.uc_sigmask = 0x1234; context.uc_onstack = 1;
	const auto before = capture_arm64_context(context);
	require(before && before->x[29] == machine.__ss.__fp && before->x[30] == machine.__ss.__lr && before->v[31] != before->v[30], "Darwin ARM context confused FP/LR/NEON fields");
	auto after = *before;
	for (auto& x : after.x) x ^= 0x8badf00d12345678;
	after.sp += 16; after.pc += 4;
	for (auto& v : after.v) for (auto& b : v) b ^= std::byte{0x91};
	require(publish_arm64_context(context, *before, after) && capture_arm64_context(context) == after, "Darwin ARM full register reply did not roundtrip");
	require(machine.__es.__far == 0x1122334455667788 && machine.__es.__esr == 0x87654321 && machine.__ss.__pad == 0x99aabbcc && context.uc_sigmask == 0x1234 && context.uc_onstack == 1, "Darwin ARM reply changed exception/signal/padding metadata");
	const auto saved = machine;
	require(!publish_arm64_context(context, *before, after) && std::memcmp(&machine, &saved, sizeof(machine)) == 0, "Darwin ARM stale reply changed context");
	context.uc_mcsize = sizeof(machine) - 1; require(!capture_arm64_context(context) && !publish_arm64_context(context, after, *before), "Darwin ARM short context accepted");
	context.uc_mcontext = nullptr; require(!capture_arm64_context(context), "Darwin ARM null context accepted");
	std::cout << "ARM Darwin context:all X/FP/LR/SP/PC/NEON/status fields;exact reply identity;exception/signal/padding preserved;synthetic context only\n";
}
} // namespace
void test_arm64_memory_instructions()
{
	test_decode(); test_refusal_and_ownership(); test_context(); test_actual_fault_context();
}
#ifdef METAL4_ARM64_MEMORY_STANDALONE
int main()
{
	try { test_arm64_memory_instructions(); }
	catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
#endif
