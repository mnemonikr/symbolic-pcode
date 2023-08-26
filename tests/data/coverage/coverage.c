/**
 * Test BOOL_NEGATE, BOOL_AND, BOOL_OR, BOOL_XOR
 */
int boolean_tests();

/**
 * Tests INT_SLESS, INT_SLESSEQUAL
 */
int signed_integer_comparison();

/**
 * Tests INT_LESS, INT_LESSEQUAL, INT_EQUAL, INT_NOTEQUAL
 */
int unsigned_integer_comparison();

/**
 * Tests INT_XOR, INT_OR, INT_AND, INT_NEGATE, INT_LEFT, INT_RIGHT, INT_SRIGHT
 */
int bitwise_arithmetic();

/**
 * Tests INT_ADD, INT_SUB, INT_MULT, INT_DIV, INT_REM, INT_ZEXT
 */
int unsigned_integer_arithmetic();

/**
 * Tests INT_ADD, INT_SUB, INT_MULT, INT_SDIV, INT_SREM, INT_SEXT
 */
int signed_integer_arithmetic();

int unsigned_integer_overflow();

int signed_integer_overflow();

int popcount();

int control_flow(int(*callback)());

int test() {
    int result;

    result = boolean_tests();
    if (result != 0) {
        return 0x10 | result;
    }

    result = signed_integer_comparison();
    if (result != 0) {
        return 0x20 | result;
    }

    result = unsigned_integer_comparison();
    if (result != 0) {
        return 0x30 | result;
    }

    result = bitwise_arithmetic();
    if (result != 0) {
        return 0x40 | result;
    }

    result = unsigned_integer_arithmetic();
    if (result != 0) {
        return 0x50 | result;
    }

    result = signed_integer_arithmetic();
    if (result != 0) {
        return 0x60 | result;
    }

    result = unsigned_integer_overflow();
    if (result != 0) {
        return 0x70 | result;
    }

    result = signed_integer_overflow();
    if (result != 0) {
        return 0x80 | result;
    }

    result = popcount();
    if (result != 0) {
        return 0x90 | result;
    }

    result = control_flow(popcount);
    if (result != 0) {
        return 0xA0 | result;
    }

    return 0;
}


int boolean_tests() {
    int x = 1 || 1;
    int y = 0 && 0;
    if (y || y) {
        return 1;
    }

    if (x && y) {
        return 2;
    }

    if ((y == 0) != (x == 1)) {
        return 3;
    }

    if (!x) {
        return 4;
    }

    return 0;
}

int signed_integer_comparison() {
    signed int x = -32;
    signed int y = -45;
    if (x < y) {
        return 1;
    }

    if (x <= y) {
        return 2;
    }

    return 0;
}

int unsigned_integer_comparison() {
    unsigned int x = 45;
    unsigned int y = 32;
    unsigned int z = 45;
    if (x < y) {
        return 1;
    }

    if (x <= y) {
        return 2;
    }

    if (x == y) {
        return 3;
    }

    if (x != z) {
        return 4;
    }

    return 0;
}

int bitnot(int a) {
    return ~a;
}


/**
 * Tests INT_XOR, INT_OR, INT_AND, INT_NEGATE, INT_LEFT, INT_RIGHT, INT_SRIGHT
 */
int bitwise_arithmetic() {
    unsigned int x = 0x5A5A5A5A;
    unsigned int y = x ^ x;

    if (y != 0) {
        return 1;
    }

    if ((y | x) != x) {
        return 2;
    }

    if ((x & x) != x) {
        return 3;
    }

    if (bitnot(0x8) != 0xFFFFFFF7) {
        return 4;
    }

    if ((x >> 16) != 0x5A5A) {
        return 5;
    }

    if ((x << 16) != 0x5A5A0000) {
        return 6;
    }

    signed int z = 0x80000000;
    if ((z >> 3) != 0xF0000000) {
        return 7;
    }

    return 0;
}

int unsigned_integer_arithmetic() {
    unsigned int x = 3;
    unsigned short y = 5;
    if ((x + y) != 8) {
        return 1;
    }

    if ((y - x) != 2) {
        return 2;
    }

    if ((x * y) != 15) {
        return 3;
    }

    if ((y / x) != 1) {
        return 4;
    }

    if ((y % x) != 2) {
        return 5;
    }

    return 0;
}

int signed_integer_arithmetic() {
    signed int x = -3;
    signed short y = -5;
    if ((x + y) != -8) {
        return 1;
    }

    if ((x - y) != 2) {
        return 2;
    }

    if ((x * y) != 15) {
        return 3;
    }

    if ((y / x) != 1) {
        return 4;
    }

    if ((y % x) != -2) {
        return 5;
    }

    if (-x != 3) {
        return 6;
    }

    return 0;
}

int unsigned_integer_overflow() {
    unsigned short x = 0xFFFF;
    unsigned short y = 1;
    unsigned short z;
    if (!__builtin_add_overflow(x, y, &z)) {
        return 1;
    }

    return 0;
}

int signed_integer_overflow() {
    signed int x = 0x80000000;
    signed int y = 1;
    signed int z;

    if (!__builtin_sadd_overflow(x, -y, &z)) {
        return 1;
    }

    if (!__builtin_ssub_overflow(x, y, &z)) {
        return 2;
    }

    return 0;
}

int control_flow(int(*callback)()) {
    __label__ control_flow_success;
    __label__ control_flow_failure;
    void* ret;

    switch (callback()) {
        case 0:
            ret = &&control_flow_success;
            break;
        case 1:
            ret = &&control_flow_failure;
            break;
        default:
            ret = &&control_flow_failure;
            break;
    }

    goto *ret;

control_flow_success:
    return 0;

control_flow_failure:
    return 1;
}

/**
 * Requires passing -mpopcnt to gcc
 */
int popcount() {
    int x = 0xFFFFFFFF;
    if (__builtin_popcount(x) != 32) {
        return 1;
    }

    return 0;
}

int main() {
    return test();
}
