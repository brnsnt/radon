import ast
import sys
import textwrap
import unittest

import pytest
from radon.visitors import *
import inspect

dedent = lambda code: textwrap.dedent(code).strip()


class assert_complexity(object):
    """Decorator that compares the cognitive complexity
    of the wrapped function to the given argument.

    Do not use with function syntax thats not supported
    in older versions (e.g. async)."""

    def __init__(self, complexity):
        self.expected = complexity

    def __call__(self, f):
        def wrapped_f(*args):
            source = inspect.getsource(f)
            tree = ast.parse(dedent(source))

            visitor = CognitiveComplexityVisitor.from_ast(tree)

            assert len(visitor.classes) == 0
            assert len(visitor.functions) == 1

            assert GET_COMPLEXITY(visitor.functions[0]) is self.expected

        return wrapped_f


class IfElseTestCase(unittest.TestCase):
    """
    Loop structures cause a structural increment (1 + nesting level),
    the else a hybrid increment (1).
    Both increase the nesting count as such.
    """

    @assert_complexity(4)
    def test_ifelse(self):

        if True:  # +1
            pass
        elif True or True:  # + 2
            pass
        else:  # + 1
            pass

    @assert_complexity(13)
    def test_if_nested(self):
        if True:  # + 1
            if True:  # + 2 (nesting=1)
                pass
            else:  # + 1
                x = 1  # To prevent parsing it as elif
                if True:  # +3 (nesting=2)
                    pass
                elif x:  # +1
                    if True:  # + 4 (nesting=3)
                        pass
                else:  # +1
                    pass

    @assert_complexity(27)
    def test_ifelse_nested(self):
        if True:  # + 1
            if True:  # + 2 (nesting=1)
                pass
            else:  # + 1
                x = 1  # To prevent parsing it as elif
                if True:  # +3 (nesting=2)
                    pass
                elif x:  # +1
                    if True:  # + 4 (nesting=3)
                        pass
                else:  # +1
                    pass
        elif True:  # + 1
            if True:  # + 2 (nesting=1)
                pass
            elif True:  # + 1
                if True:  # + 3
                    pass
        else:  # + 1

            x = 1
            if True:  # + 2 (nesting=1)
                pass
            elif True:  # + 1
                x = 1

                if True:  # + 3
                    pass

    @assert_complexity(7)
    def test_boolean_sequences(self):
        """Asses for each sequence of binary logical operators,
        successive 'and' or 'or' increment the complexity only by 1."""

        a, b, c, d = True, True, False, False

        if a and b and c or b and c and d:  # +1 + 3
            pass
        elif a or b or c and d:  # +1 +2
            pass

    @assert_complexity(6)
    def test_if_expression(self):
        """
        If Expressions get only one hybrid increment for the else.
        The 'if' is not subject to a structural increment but adds
        a nesting level.
        """
        a = 1 if True else 0  # + 1
        b = 1 if True and True else 0  # + 2
        c = 1 if True and True and True or False else 0  # + 3

    @assert_complexity(9)
    def test_if_expression_nested(self):
        """See test_if_expression.
        Test that the nesting levels apply"""

        a, b, c, d = True, True, False, False

        a = 1 if (a if b else c) else 0  # + 1 + 2
        a = 1 if (a if b else (b if c else 0)) else 0  # + 1 + 2 + 3


class LoopsTestCase(unittest.TestCase):
    """
    Loop structures cause a structural increment (1 + nesting level),
    the else clauses a hybrid increment (nesting level not counted).
    Both increase the nesting count.
    """

    @assert_complexity(3)
    def test_for_else(self):
        for x in range(10):  # +1
            pass

        for x in range(10):  # +1
            pass
        else:
            pass  # + 1

    @assert_complexity(6)
    def test_for_nested(self):
        for x in range(10):  # +1
            for y in range(10):  # +2 (nesting=1)
                pass

        else:  # + 1
            for y in range(5):  # +2
                pass

    @assert_complexity(15)
    def test_for_deeply_nested(self):
        for x in range(10):  # +1
            for y in range(10):  # +2 (nesting=1)
                for y in range(10):  # +3 (nesting=2)
                    pass
                else:  # +1
                    pass
        else:  # + 1
            for y in range(5):  # +2 (nesting=1)
                for y in range(10):  # +3 (nesting=2)
                    pass
                else:  # +1
                    pass
            else:  # +1
                pass

    @assert_complexity(2)
    def test_while(self):
        while 5 < 4:  # + 1
            pass
        else:  # + 1
            pass

    @assert_complexity(9)
    def test_while_nested(self):
        while True and False:  # + 2
            while False:  # + 2 (nesting=1)
                pass
            else:  # + 1
                pass
        else:  # + 1
            while False:  # +2 (nesting=1)
                pass
            else:  # +1
                pass

    @assert_complexity(17)
    def test_while_deeply_nested(self):
        while True and False:  # + 2
            while False:  # + 2 (nesting=1)
                while False:  # + 3 (nesting=2)
                    pass
            else:  # + 1
                pass
        else:  # + 1
            while False:  # + 2 (nesting=1)
                while False and False and False or True:  # + 3 (nesting=2) + 2
                    pass
                else:  # + 1
                    pass

    @assert_complexity(4)
    def test_with(self):
        """The context manager itself does not
        add an increment but increases the nesting level.

        See it at method-like-structure similar to"""

        with open("raw.py") as fobj:
            print(fobj.read())

        with open("raw.py") as fobj:
            if True:  # +2 for nesting
                pass

            for a in range(5):  # + 2
                pass


class GeneratorTestCase(unittest.TestCase):
    @assert_complexity(7)
    def test_generators(self):
        """Generators add a structural increment,
        and it's if clauses a hybrid increment."""

        g = [i for i in range(4)]  # + 1
        g = [i for i in range(4) if i & 1]  # + 2

        g = (i for i in range(4))  # + 1
        g = (i for i in range(4) if i & 1)  # + 2

        s = sum(i for i in range(10))  # + 1

    @assert_complexity(10)
    def test_generators_nested(self):
        """A nested generator inside an if clause
        should get a nesting increment.

        But if clauses on nested generators adds
        as hybrid increment only 1."""

        # 1 + 1 + 2
        g = [i for i in range(42) if sum(k ** 2 for k in divisors(i)) & 1]

        # 1 for + 1 if + (2 for + 1 if + 1 Bool)
        g = (
            i for i in range(42) if sum(k ** 2 for k in divisors(i) if k > 10 and True)
        )

    @assert_complexity(7)
    def test_generators_double(self):
        """Each generator and if increment the complexity by 1.
        """

        # 1 + 1 + 1
        g = sum(i for i in range(12) for z in range(i ** 2) if i * z & 1)

        # 1 + 3
        g = sum(i for i in range(10) if i >= 2 and val and val2 or val3)

    @assert_complexity(2)
    def test_lambda(self):
        """Lambda expressions do not increment
        the complexity but add a nesting level."""

        k = lambda a, b: k(a, b)

        k = lambda a, b, c: c if a else b  # + 2

    @assert_complexity(3)
    def test_set_generator(self):
        # => 2.7
        x = {i for i in range(4)}  # + 1
        x = {i for i in range(4) if i & 1}  # + 2

    @assert_complexity(3)
    def test_dict_generator(self):

        x = {i: i ** 4 for i in range(4)}  # + 1
        x = {i: i ** 4 for i in range(4) if i & 1}  # + 2


class TryExceptTestCase(unittest.TestCase):
    @assert_complexity(2)
    def test_try_except(self):
        try:
            raise TypeError
        except TypeError:  # + 1
            pass
        else:  # + 1
            pass

    @assert_complexity(13)
    def test_try_full(self):

        try:
            if True:  # + 2 (nesting=1)
                raise TypeError
        except TypeError:  # + 1
            if True:  # + 2 (nesting=1)
                raise TypeError
        except TypeError:  # + 1
            if True:  # + 2 (nesting=1)
                raise TypeError
        else:  # + 1
            if True:  # + 2 (nesting=1)
                raise TypeError
        finally:
            if True:  # + 2 (nesting=1)
                raise TypeError

    @assert_complexity(19)
    def test_try_nested(self):

        try:
            if True:  # + 2 (nesting=1)
                raise TypeError
        except TypeError:  # + 1
            try:
                if True and True:  # + 4 (nesting=2)
                    pass
                else:  # + 1
                    pass
            except ValueError:  # + 1
                pass
            else:  # + 1
                if True:  # + 3 (nesting=3)
                    pass
        except TypeError:  # + 1
            pass
        else:  # + 1
            if True:  # + 2 (nesting=2)
                pass
        finally:
            if True:  # + 2 (nesting=2)
                pass


class MixedFunctionsTestCase(unittest.TestCase):
    a, b, c = 1, 2, 3
    if a and not b:
        pass
    elif b or c:
        pass
    else:
        pass

    for i in range(4):
        pass

    @assert_complexity(8)
    def test_mixed_function(self):
        while a < b:  # + 1
            b, a = a ** 2, b ** 2

        a, b, c = 1, 2, 3
        if a and b == 4:  # + 2
            return c ** c
        elif a and not c:  # + 2
            return sum(i for i in range(41) if i & 1)  # + 3
        return a + b


SINGLE_FUNCTIONS_CASES = []
if sys.version_info[:2] >= (3, 5):
    # With and async-with statements no longer count towards CC, see #123
    SINGLE_FUNCTIONS_CASES = [
        (
            """
         async def f(a, b):
            async with open('blabla.log', 'w') as f:
                async for i in range(100):
                    f.write(str(i) + '\\n')
         """,
            (0, 2),
        ),
    ]


@pytest.mark.parametrize("code,expected", SINGLE_FUNCTIONS_CASES)
def test_visitor_single_functions(code, expected):
    visitor = CognitiveComplexityVisitor.from_code(dedent(code))
    assert len(visitor.functions) == 1
    assert (visitor.complexity, visitor.functions[0].complexity) == expected


class ClassComplexityTestCase(unittest.TestCase):
    if True and True or False:  # + 3
        pass

    for a in range(10):  # + 1
        pass

    def a_func(self):
        if True:  # + 1
            pass

    def some_func(self):
        if True:  # + 1
            pass
        elif a and b:  # + 2
            pass


def test_class_visit():
    """
    The total complexity of a class is
    the sum of its function and body complexity.

    The number of functions and the class itself
    dont add an increment.
    """
    total_class_complexity = 8
    methods_complexity = (1, 3)

    source = inspect.getsource(ClassComplexityTestCase)
    tree = ast.parse(dedent(source))

    visitor = CognitiveComplexityVisitor.from_ast(tree)

    assert len(visitor.classes) == 1
    assert len(visitor.functions) == 0
    cls = visitor.classes[0]
    assert cls.real_complexity == total_class_complexity
    assert tuple(map(GET_COMPLEXITY, cls.methods)) == methods_complexity


def test_module_visit():
    source = """
        class A(object):
            if True: # + 1
                pass

        class B(object):
            def f(self):
                for a in range(10): # + 1
                    pass
        """


######################
# TODO cleanup

SIMPLE_BLOCKS = [
    (
        """
     for x in range(10): print(x)
     """,
        1,
        {},
    ),
    (
        """
     for x in xrange(10): print(x)
     else: pass
     """,
        2,
        {},
    ),
    (
        """
     while a < 4: pass
     """,
        1,
        {},
    ),
    (
        """
     while a < 4: pass
     else: pass
     """,
        2,
        {},
    ),
    (
        """
     while a < 4 and b < 42: pass
     """,
        2,
        {},
    ),
    (
        """
     while a and b or c < 10: pass
     else: pass
     """,
        4,
        {},
    ),
    # With and async-with statements no longer count towards CC, see #123
    (
        """
     with open('raw.py') as fobj: print(fobj.read())
     """,
        0,
        {},
    ),
    # Nesting increment for nested methods and similar
    (
        """
     with open('raw.py') as fobj:
        if True:                       # +2 for nesting
          print(fobj.read()) 
     """,
        2,
        {},
    ),
    (
        """
     [i for i in range(4)]
     """,
        1,
        {},
    ),
    (
        """
     [i for i in range(4) if i&1]
     """,
        2,
        {},
    ),
    (
        """
     (i for i in range(4))
     """,
        1,
        {},
    ),
    (
        """
     (i for i in range(4) if i&1)
     """,
        2,
        {},
    ),
    # Nesting increment for inner
    (
        """
     [i for i in range(42) if sum(k ** 2 for k in divisors(i)) & 1]
     """,
        4,
        {},
    ),
    (
        """
     try: raise TypeError
     except TypeError: pass
     """,
        1,  # 1 except
        {},
    ),
    (
        """
     try: raise TypeError
     except TypeError: pass
     else: pass
     """,
        2,  # 1 except, 1 else
        {},
    ),
    (
        """
     try: raise TypeError
     finally: pass
     """,
        0,  # completely ignored
        {},
    ),
    (
        """
     try: raise TypeError
     except TypeError: pass
     finally: pass
     """,
        1,  # except
        {},
    ),
    (
        """
     try: raise TypeError
     except TypeError: pass # + 1
     else: pass             # + 1
     finally: pass
     """,
        2,
        {},
    ),
    # Lambda are not counted
    (
        """
     k = lambda a, b: k(b, a)
     """,
        0,
        {},
    ),
    # Lambda add a nesting increment to inner control structures
    (
        """
     k = lambda a, b, c: c if a else b
     """,
        2,  # if: 2 (nesting=1), else: 1 (nesting not added)
        {},
    ),
    # Not clear from definition but shorthand are encouraged.
    (
        """
     v = a if b else c
     """,
        1,
        {},
    ),
    (
        """
     v = a if sum(i for i in xrange(c)) < 10 else c
     """,
        3,  # nested generator +2 , + 1 for else
        {},
    ),
    # TODO maybe better count second for as nested
    (
        """
     sum(i for i in range(12) for z in range(i ** 2) if i * z & 1)
     """,
        3,  # 1 for each for, 1 if, 1 for each *,&
        {},
    ),
    (
        """
     sum(i for i in range(10) if i >= 2 and val and val2 or val3)
     """,
        4,
        {},
    ),
    (
        """
     for i in range(10):
         print(i)
     else:
         print('wah')
         print('really not found')
         print(3)
     """,
        2,
        {},
    ),
    (
        """
     while True:
         print(1)
     else:
         print(2)
         print(1)
         print(0)
         print(-1)
     """,
        2,
        {},
    ),
    (
        """
     assert i < 0
     """,
        1,
        {},
    ),
    (
        """
     assert i < 0, "Fail"
     """,
        1,
        {},
    ),
    (
        """
     assert i < 0
     """,
        0,
        {"no_assert": True},
    ),
    (
        """
     def f():
        assert 10 > 20
     """,
        0,
        {"no_assert": True},
    ),
    (
        """
     class TestYo(object):
        def test_yo(self):
            assert self.n > 4
     """,
        0,
        {"no_assert": True},
    ),
    (
        """
     try:
         1 / 0
     except ZeroDivisonError:         # +1
         print
     except TypeError:                # +1 
         pass
        """,
        2,
        {},
    ),
]


# These run only if Python version is >= 2.7
ADDITIONAL_BLOCKS = [
    (
        """
     {i for i in range(4)}
     """,
        1,
        {},
    ),
    (
        """
     {i for i in range(4) if i&1}
     """,
        2,
        {},
    ),
    (
        """
     {i:i**4 for i in range(4)}
     """,
        1,
        {},
    ),
    (
        """
     {i:i**4 for i in range(4) if i&1}
     """,
        2,
        {},
    ),
    # TODO unclear
    # Gives nesting level 3 for comprehension when counting function calls
    # as nesting as well.
    (
        """
        while self.m(k) < k:          # +1
            k -= m(k ** 2 - min(m(j) for j in range(k ** 4))) # +3 (nesting=1)
            return k
        """,
        3,
        {},
    ),
]

BLOCKS = SIMPLE_BLOCKS[:]
if sys.version_info[:2] >= (2, 7):
    BLOCKS.extend(ADDITIONAL_BLOCKS)


@pytest.mark.parametrize("code,expected,kwargs", BLOCKS)
def test_visitor_simple(code, expected, kwargs):
    _test_visitor_simple(code, expected, kwargs)


def _test_visitor_simple(code, expected, kwargs):
    visitor = CognitiveComplexityVisitor.from_code(dedent(code), **kwargs)
    assert visitor.complexity == expected


SINGLE_FUNCTIONS_CASES = [
    (
        """
     def f(a, b, c):
        if a and b == 4:
            return c ** c
        elif a and not c:
            return sum(i for i in range(41) if i&1)
        return a + b
     """,
        (0, 7),
    ),
    (
        """
     if a and not b: pass
     elif b or c: pass
     else: pass

     for i in range(4):
        print(i)

     def g(a, b):
        while a < b:
            b, a = a **2, b ** 2
        return b
     """,
        (6, 1),
    ),
    (
        """
     def f(a, b):
        while a**b:                  # +1
            a, b = b, a * (b - 1)
            if a and b:              # + 3
                b = 0
            else:                    # + 1
                b = 1
        return sum(i for i in range(b)) # +1
     """,
        (0, 6),
    ),
]

if sys.version_info[:2] >= (3, 5):
    # With and async-with statements no longer count towards CC, see #123
    SINGLE_FUNCTIONS_CASES.append(
        (
            """
         async def f(a, b):
            async with open('blabla.log', 'w') as f:
                async for i in range(100):
                    f.write(str(i) + '\\n')
         """,
            (0, 2),
        ),
    )


@pytest.mark.parametrize("code,expected", SINGLE_FUNCTIONS_CASES)
def test_visitor_single_functions(code, expected):
    visitor = CognitiveComplexityVisitor.from_code(dedent(code))
    assert len(visitor.functions) == 1
    assert (visitor.complexity, visitor.functions[0].complexity) == expected


FUNCTIONS_CASES = [
    # With and async-with statements no longer count towards CC, see #123
    (
        """
     def f(a, b):
        return a if b else 2              # +1

     def g(a, b, c):
        if a and b:                       # +2
            return a / b + b / a
        elif b and c:                     # +2
            if a:                         # +2 (nesting=1)
                return b
            else:                         # +1
                return b / c - c / b
        return a + b + c

     def h(a, b):
        return 2 * (a + b)
     """,
        (1, 7, 0),
    ),
    (
        """
     def f(p, q):
        while p:                      # +1
            p, q = q, p - q
        if q < 1:                     # +1
            return 1 / q ** 2
        elif q > 100:                 # +1
            return 1 / q ** .5
        return 42 if not q else p     # +1

     def g(a, b, c):
        if a and b or a - b:          # +3
            return a / b - c
        elif b or c:                  # +2
            return 1
        else:                         # +1
            k = 0
            with open('results.txt', 'w') as fobj:
                for i in range(b ** c):                              # +3 (nesting=2)
                    k += sum(1 / j for j in range(i ** 2) if j > 2)  # +5 (nesting=3) 
                fobj.write(str(k))
            return k - 1
     """,
        (4, 14),
    ),
]


@pytest.mark.parametrize("code,expected", FUNCTIONS_CASES)
def test_visitor_functions(code, expected):
    visitor = CognitiveComplexityVisitor.from_code(dedent(code))
    assert len(visitor.functions) == len(expected)
    assert tuple(map(GET_COMPLEXITY, visitor.functions)) == expected


CLASSES_CASES = [
    (
        """
     class A(object):

         def m(self, a, b):
             if not a or b:                # +2
                 return b - 1
             try:
                 return a / b
             except ZeroDivisionError:     # +1
                 return a

         def n(self, k):
             while self.m(k) < k:          # +1
                 k -= self.m(k ** 2 - min(self.m(j) for j in range(k ** 4))) # +2
             return k
     """,
        (6, 3, 3),
    ),
    (
        """
     class B(object):

         ATTR = 9 if A().n(9) == 9 else 10                     # +1
         import sys
         if sys.version_info >= (3, 3):                        # +1
             import os
             AT = os.openat('/random/loc')

         def __iter__(self):
             return __import__('itertools').tee(B.__dict__)

         def test(self, func):
             a = func(self.ATTR, self.AT)
             if a < self.ATTR:                                # +1
                 yield self
             elif a > self.ATTR ** 2:                         # +1
                 yield self.__iter__()
             yield iter(a)
     """,
        (4, 0, 2),
    ),
    (
        """
    class J(object):

         def aux(self, w):           
             if w == 0:               # +1
                 return 0
             return w - 1 + sum(self.aux(w - 3 - i) for i in range(2))  # +1
    """,
        (2, 2),
    ),
]


@pytest.mark.parametrize("code,expected", CLASSES_CASES)
def test_visitor_classes(code, expected):
    total_class_complexity = expected[0]
    methods_complexity = expected[1:]
    visitor = CognitiveComplexityVisitor.from_code(dedent(code))
    assert len(visitor.classes) == 1
    assert len(visitor.functions) == 0
    cls = visitor.classes[0]
    assert cls.real_complexity == total_class_complexity
    assert tuple(map(GET_COMPLEXITY, cls.methods)) == methods_complexity


GENERAL_CASES = [
    (
        """
     if a and b:  #+2
         print
     else:        #+1
         print
     a = sum(i for i in range(1000) if i % 3 == 0 and i % 5 == 0) #+3

     def f(n):
         def inner(n):
             return n ** 2

         if n == 0:     # +1
             return 1
         elif n == 1:   #+1
             return n
         elif n < 5:    #+1
             return (n - 1) ** 2
         return n * pow(inner(n), f(n - 1), n - 3)
     """,
        (6, 4, 0, 10),
    ),
    (
        """
     try:
         1 / 0
     except ZeroDivisonError:         # +1
         print
     except TypeError:                # +1 
         pass

     class J(object):

         def aux(self, w):
             if w == 0:               # +1
                 return 0
             return w - 1 + sum(self.aux(w - 3 - i) for i in range(2))  # +1

     def f(a, b):
         def inner(n):
             return n ** 2
         if a < b:                # +1
             b, a = a, inner(b)
         return a, b
     """,
        (3, 1, 2, 5),
    ),
    # TODO does this make sense?
    (
        """
     class f(object):
         class inner(object):
             pass
     """,
        (0, 0, 0, 0),
    ),
]


@pytest.mark.parametrize("code,expected", GENERAL_CASES)
def test_visitor_module(code, expected):
    (
        module_complexity,
        functions_complexity,
        classes_complexity,
        total_complexity,
    ) = expected

    visitor = CognitiveComplexityVisitor.from_code(dedent(code))
    assert visitor.complexity or module_complexity is not None
    assert visitor.functions_complexity == functions_complexity
    assert visitor.classes_complexity == classes_complexity
    assert visitor.total_complexity == total_complexity


CLOSURES_CASES = [
    (
        """
     def f(n):
         def g(l):
             return l ** 4
         def h(i): # nesting level now at 1
             return i ** 5 + 1 if i & 1 else 2                    #+2
         return sum(g(u + 4) / float(h(u)) for u in range(2, n))  #+1
     """,
        ("g", "h"),
        (0, 2, 3),
    ),
    (
        """
     # will it work? :D
     def memoize(func):
         cache = {}
         def aux(*args, **kwargs): # + 0 but nesting level at 1
             key = (args, kwargs)
             if key in cache:  # + 2
                 return cache[key]
             cache[key] = res = func(*args, **kwargs)
             return res
         return aux
     """,
        ("aux",),
        (2, 2),
    ),
]


@pytest.mark.parametrize("code,closure_names,expected", CLOSURES_CASES)
def test_visitor_closures(code, closure_names, expected):
    visitor = CognitiveComplexityVisitor.from_code(dedent(code))
    func = visitor.functions[0]
    closure_names = closure_names
    expected_cs_cc = expected[:-1]
    expected_total_cc = expected[-1]

    assert len(visitor.functions) == 1

    names = tuple(cs.name for cs in func.closures)
    assert names == closure_names

    cs_complexity = tuple(cs.complexity for cs in func.closures)
    assert cs_complexity == expected_cs_cc
    assert func.complexity == expected_total_cc

    # There was a bug for which `blocks` increased while it got accessed
    v = visitor
    assert v.blocks == v.blocks == v.blocks


CONTAINERS_CASES = [
    (("func", 12, 0, 18, False, None, [], 5), ("F", "func", "F 12:0->18 func - 5")),
    (
        ("meth", 12, 0, 21, True, "cls", [], 5),
        ("M", "cls.meth", "M 12:0->21 cls.meth - 5"),
    ),
    (("cls", 12, 0, 15, [], [], 5), ("C", "cls", "C 12:0->15 cls - 5")),
    (
        ("cls", 12, 0, 19, [object, object, object, object], [], 30),
        ("C", "cls", "C 12:0->19 cls - 8"),
    ),
]


@pytest.mark.parametrize("values,expected", CONTAINERS_CASES)
def test_visitor_containers(values, expected):
    expected_letter, expected_name, expected_str = expected

    cls = Function if len(values) == 8 else Class
    obj = cls(*values)
    assert obj.letter == expected_letter
    assert obj.fullname == expected_name
    assert str(obj) == expected_str
