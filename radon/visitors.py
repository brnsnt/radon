'''This module contains the ComplexityVisitor class which is where all the
analysis concerning Cyclomatic Complexity is done. There is also the class
HalsteadVisitor, that counts Halstead metrics.'''

import ast
import operator
import collections


# Helper functions to use in combination with map()
GET_COMPLEXITY = operator.attrgetter('complexity')
GET_REAL_COMPLEXITY = operator.attrgetter('real_complexity')
NAMES_GETTER = operator.attrgetter('name', 'asname')
GET_ENDLINE = operator.attrgetter('endline')

BaseFunc = collections.namedtuple('Function', ['name', 'lineno', 'col_offset',
                                               'endline', 'is_method',
                                               'classname', 'closures',
                                               'complexity'])
BaseClass = collections.namedtuple('Class', ['name', 'lineno', 'col_offset',
                                             'endline', 'methods',
                                             'inner_classes',
                                             'real_complexity'])


def code2ast(source):
    '''Convert a string object into an AST object.

    This function is retained for backwards compatibility, but it no longer
    attemps any conversions. It's equivalent to a call to ``ast.parse``.
    '''
    return ast.parse(source)


class Function(BaseFunc):
    '''Object represeting a function block.'''

    @property
    def letter(self):
        '''The letter representing the function. It is `M` if the function is
        actually a method, `F` otherwise.
        '''
        return 'M' if self.is_method else 'F'

    @property
    def fullname(self):
        '''The full name of the function. If it is a method, then the full name
        is:
                {class name}.{method name}
        Otherwise it is just the function name.
        '''
        if self.classname is None:
            return self.name
        return '{0}.{1}'.format(self.classname, self.name)

    def __str__(self):
        '''String representation of a function block.'''
        return '{0} {1}:{2}->{3} {4} - {5}'.format(self.letter, self.lineno,
                                                   self.col_offset,
                                                   self.endline,
                                                   self.fullname,
                                                   self.complexity)


class Class(BaseClass):
    '''Object representing a class block.'''

    letter = 'C'

    @property
    def fullname(self):
        '''The full name of the class. It is just its name. This attribute
        exists for consistency (see :data:`Function.fullname`).
        '''
        return self.name

    @property
    def complexity(self):
        '''The average complexity of the class. It corresponds to the average
        complexity of its methods plus one.
        '''
        if not self.methods:
            return self.real_complexity
        methods = len(self.methods)
        return int(self.real_complexity / float(methods)) + (methods > 1)

    def __str__(self):
        '''String representation of a class block.'''
        return '{0} {1}:{2}->{3} {4} - {5}'.format(self.letter, self.lineno,
                                                   self.col_offset,
                                                   self.endline, self.name,
                                                   self.complexity)


class CodeVisitor(ast.NodeVisitor):
    '''Base class for every NodeVisitors in `radon.visitors`. It implements a
    couple utility class methods and a static method.
    '''

    @staticmethod
    def get_name(obj):
        '''Shorthand for ``obj.__class__.__name__``.'''
        return obj.__class__.__name__

    @classmethod
    def from_code(cls, code, **kwargs):
        '''Instanciate the class from source code (string object). The
        `**kwargs` are directly passed to the `ast.NodeVisitor` constructor.
        '''
        return cls.from_ast(code2ast(code), **kwargs)

    @classmethod
    def from_ast(cls, ast_node, **kwargs):
        '''Instantiate the class from an AST node. The `**kwargs` are
        directly passed to the `ast.NodeVisitor` constructor.
        '''
        visitor = cls(**kwargs)
        visitor.visit(ast_node)
        return visitor


class ComplexityVisitor(CodeVisitor):
    '''A visitor that keeps track of the cyclomatic complexity of
    the elements.

    :param to_method: If True, every function is treated as a method. In this
        case the *classname* parameter is used as class name.
    :param classname: Name of parent class.
    :param off: If True, the starting value for the complexity is set to 1,
        otherwise to 0.
    '''

    def __init__(self, to_method=False, classname=None, off=True,
                 no_assert=False):
        self.off = off
        self.complexity = 1 if off else 0
        self.functions = []
        self.classes = []
        self.to_method = to_method
        self.classname = classname
        self.no_assert = no_assert
        self._max_line = float('-inf')

    @property
    def functions_complexity(self):
        '''The total complexity from all functions (i.e. the total number of
        decision points + 1).

        This is *not* the sum of all the complexity from the functions. Rather,
        it's the complexity of the code *inside* all the functions.
        '''
        return sum(map(GET_COMPLEXITY, self.functions)) - len(self.functions)

    @property
    def classes_complexity(self):
        '''The total complexity from all classes (i.e. the total number of
        decision points + 1).
        '''
        return sum(map(GET_REAL_COMPLEXITY, self.classes)) - len(self.classes)

    @property
    def total_complexity(self):
        '''The total complexity. Computed adding up the visitor complexity, the
        functions complexity, and the classes complexity.
        '''
        return (self.complexity + self.functions_complexity +
                self.classes_complexity + (not self.off))

    @property
    def blocks(self):
        '''All the blocks visited. These include: all the functions, the
        classes and their methods. The returned list is not sorted.
        '''
        blocks = []
        blocks.extend(self.functions)
        for cls in self.classes:
            blocks.append(cls)
            blocks.extend(cls.methods)
        return blocks

    @property
    def max_line(self):
        '''The maximum line number among the analyzed lines.'''
        return self._max_line

    @max_line.setter
    def max_line(self, value):
        '''The maximum line number among the analyzed lines.'''
        if value > self._max_line:
            self._max_line = value

    def generic_visit(self, node):
        '''Main entry point for the visitor.'''
        # Get the name of the class
        name = self.get_name(node)
        # Check for a lineno attribute
        if hasattr(node, 'lineno'):
            self.max_line = node.lineno
        # The Try/Except block is counted as the number of handlers
        # plus the `else` block.
        # In Python 3.3 the TryExcept and TryFinally nodes have been merged
        # into a single node: Try
        if name in ('Try', 'TryExcept'):
            self.complexity += len(node.handlers) + len(node.orelse)
        elif name == 'BoolOp':
            self.complexity += len(node.values) - 1
        # Ifs, with and assert statements count all as 1.
        # Note: Lambda functions are not counted anymore, see #68
        elif name in ('If', 'IfExp'):
            self.complexity += 1
        # The For and While blocks count as 1 plus the `else` block.
        elif name in ('For', 'While', 'AsyncFor'):
            self.complexity += bool(node.orelse) + 1
        # List, set, dict comprehensions and generator exps count as 1 plus
        # the `if` statement.
        elif name == 'comprehension':
            self.complexity += len(node.ifs) + 1

        super(ComplexityVisitor, self).generic_visit(node)

    def visit_Assert(self, node):
        '''When visiting `assert` statements, the complexity is increased only
        if the `no_assert` attribute is `False`.
        '''
        self.complexity += not self.no_assert

    def visit_AsyncFunctionDef(self, node):
        '''Async function definition is the same thing as the synchronous
        one.
        '''
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        '''When visiting functions a new visitor is created to recursively
        analyze the function's body.
        '''
        # The complexity of a function is computed taking into account
        # the following factors: number of decorators, the complexity
        # the function's body and the number of closures (which count
        # double).
        closures = []
        body_complexity = 1
        for child in node.body:
            visitor = ComplexityVisitor(off=False, no_assert=self.no_assert)
            visitor.visit(child)
            closures.extend(visitor.functions)
            # Add general complexity but not closures' complexity, see #68
            body_complexity += visitor.complexity

        func = Function(node.name, node.lineno, node.col_offset,
                        max(node.lineno, visitor.max_line), self.to_method,
                        self.classname, closures, body_complexity)
        self.functions.append(func)

    def visit_ClassDef(self, node):
        '''When visiting classes a new visitor is created to recursively
        analyze the class' body and methods.
        '''
        # The complexity of a class is computed taking into account
        # the following factors: number of decorators and the complexity
        # of the class' body (which is the sum of all the complexities).
        methods = []
        # According to Cyclomatic Complexity definition it has to start off
        # from 1.
        body_complexity = 1
        classname = node.name
        visitors_max_lines = [node.lineno]
        inner_classes = []
        for child in node.body:
            visitor = ComplexityVisitor(
                True, classname, off=False, no_assert=self.no_assert
            )
            visitor.visit(child)
            methods.extend(visitor.functions)
            body_complexity += (
                visitor.complexity
                + visitor.functions_complexity
                + len(visitor.functions)
            )
            visitors_max_lines.append(visitor.max_line)
            inner_classes.extend(visitor.classes)

        cls = Class(
            classname,
            node.lineno,
            node.col_offset,
            max(visitors_max_lines + list(map(GET_ENDLINE, methods))),
            methods,
            inner_classes,
            body_complexity,
        )
        self.classes.append(cls)


class HalsteadVisitor(CodeVisitor):
    """Visitor that keeps track of operators and operands, in order to compute
    Halstead metrics (see :func:`radon.metrics.h_visit`).
    """

    # ast type name to attribute
    # As of Python 3.8 Num/Str/Bytes/NameConstat
    # are now in a common node Constant.
    types = {"Num": "n", "Name": "id", "Attribute": "attr", "Constant": "value"}

    def __init__(self, context=None):
        """*context* is a string used to keep track the analysis' context."""
        self.operators_seen = set()
        self.operands_seen = set()
        self.operators = 0
        self.operands = 0
        self.context = context

        # A new visitor is spawned for every scanned function.
        self.function_visitors = []

    @property
    def distinct_operators(self):
        '''The number of distinct operators.'''
        return len(self.operators_seen)

    @property
    def distinct_operands(self):
        '''The number of distinct operands.'''
        return len(self.operands_seen)

    def dispatch(meth):
        '''This decorator does all the hard work needed for every node.

        The decorated method must return a tuple of 4 elements:

            * the number of operators
            * the number of operands
            * the operators seen (a sequence)
            * the operands seen (a sequence)
        """

        def aux(self, node):
            """Actual function that updates the stats."""
            results = meth(self, node)
            self.operators += results[0]
            self.operands += results[1]
            self.operators_seen.update(results[2])
            for operand in results[3]:

                name = self.get_name(operand)
                new_operand = getattr(operand, self.types.get(name, ""), operand)

                self.operands_seen.add((self.context, new_operand))
            # Now dispatch to children
            super(HalsteadVisitor, self).generic_visit(node)

        return aux

    @dispatch
    def visit_BinOp(self, node):
        """A binary operator."""
        return (1, 2, (self.get_name(node.op),), (node.left, node.right))

    @dispatch
    def visit_UnaryOp(self, node):
        """A unary operator."""
        return (1, 1, (self.get_name(node.op),), (node.operand,))

    @dispatch
    def visit_BoolOp(self, node):
        """A boolean operator."""
        return (1, len(node.values), (self.get_name(node.op),), node.values)

    @dispatch
    def visit_AugAssign(self, node):
        """An augmented assign (contains an operator)."""
        return (1, 2, (self.get_name(node.op),), (node.target, node.value))

    @dispatch
    def visit_Compare(self, node):
        """A comparison."""
        return (
            len(node.ops),
            len(node.comparators) + 1,
            map(self.get_name, node.ops),
            node.comparators + [node.left],
        )

    def visit_FunctionDef(self, node):
        """When visiting functions, another visitor is created to recursively
        analyze the function's body. We also track information on the function
        itself.
        """
        func_visitor = HalsteadVisitor(context=node.name)

        for child in node.body:
            visitor = HalsteadVisitor.from_ast(child, context=node.name)
            self.operators += visitor.operators
            self.operands += visitor.operands
            self.operators_seen.update(visitor.operators_seen)
            self.operands_seen.update(visitor.operands_seen)

            func_visitor.operators += visitor.operators
            func_visitor.operands += visitor.operands
            func_visitor.operators_seen.update(visitor.operators_seen)
            func_visitor.operands_seen.update(visitor.operands_seen)

        # Save the visited function visitor for later reference.
        self.function_visitors.append(func_visitor)


class CognitiveComplexityVisitor(CodeVisitor):
    """A visitor that keeps track of the cognitive complexity of the elements.

    A Cognitive Complexity score is assessed according to three basic rules:
    1. Ignore structures that allow multiple statements to be readably shorthanded into one
    2. Increment (add one) for each break in the linear flow of the code
    3. Increment when flow-breaking structures are nested
    
    Additionally, a complexity score is made up of four different types of increments:
    A. Nesting - assessed for nesting control flow structures inside each other
       - Whenever structural/hybrid incrementing structures are nested inside each other
    B. Structural - assessed on control flow structures that are subject to a nesting
    increment, and that increase the nesting count
       - Loops: For, While, AsyncFor, AsyncWhile,
       - Conditional: If, Except
       - Generator
    C. Fundamental - assessed on statements not subject to a nesting increment
       - Sequence of binaries: BoolOp
       - Recursion
       - If in generator
    D. Hybrid - assessed on control flow structures that are not subject to a nesting
    increment, but which do increase the nesting count
       - Else, Elif, Try, With

    TODO:
      - IfExpr is unclear: Currently structural maybe better hybrid/fundamental
      - With is unclear: Currently hybrid maybe ignore 
      - Closures and Lambdas should count the nesting level
      - Handle no_assert parameter

    :param to_method: If True, every function is treated as a method. In this
        case the *classname* parameter is used as class name.
    :param classname: Name of parent class.
    :param off: If True, the starting value for the complexity is set to 1,
        otherwise to 0.
    """

    def __init__(
        self,
        to_method=False,
        classname=None,
        off=False,
        no_assert=False,
        nesting=0,
        func_name=None,
        is_elif=False,
        in_closure=False,
    ):
        self.off = off
        self.complexity = 1 if off else 0
        self.functions = []
        self.classes = []
        self.to_method = to_method
        self.classname = classname
        self.no_assert = no_assert
        self._max_line = float("-inf")
        self.nesting = nesting
        self.func_name = func_name  # required to detect recursion

        # Is the current visit a visit of an elif node ?
        # An elif appears as nested if nodes and can't be
        # distinguished otherwise as it seems.
        self.is_elif = False

        self.in_closure = in_closure

    @property
    def functions_complexity(self):
        """The total complexity from all functions (i.e. the total number of
        decision points + 1).

        This is *not* the sum of all the complexity from the functions. Rather,
        it's the complexity of the code *inside* all the functions.
        """
        return sum(map(GET_COMPLEXITY, self.functions))

    @property
    def classes_complexity(self):
        """The total complexity from all classes (i.e. the total number of
        decision points + 1).
        """
        return sum(map(GET_REAL_COMPLEXITY, self.classes))

    @property
    def total_complexity(self):
        """The total complexity. Computed adding up the visitor complexity, the
        functions complexity, and the classes complexity.
        """

        return self.complexity + self.functions_complexity + self.classes_complexity

    @property
    def blocks(self):
        """All the blocks visited. These include: all the functions, the
        classes and their methods. The returned list is not sorted.
        """
        blocks = []
        blocks.extend(self.functions)
        for cls in self.classes:
            blocks.append(cls)
            blocks.extend(cls.methods)
        return blocks

    @property
    def max_line(self):
        """The maximum line number among the analyzed lines."""
        return self._max_line

    @max_line.setter
    def max_line(self, value):
        """The maximum line number among the analyzed lines."""
        if value > self._max_line:
            self._max_line = value

    def print_node(self, name, symbol=""):
        SEP = "    " * self.nesting
        print(
            "{}".format(SEP), symbol, name, self.complexity, self.nesting,
        )

    def visit_If(self, node):
        # Get the name of the class
        name = self.get_name(node)

        # Check for a lineno attribute
        if hasattr(node, "lineno"):
            self.max_line = node.lineno

        self.complexity += 1
        if not self.is_elif:
            self.complexity += self.nesting

        # self.print_node("elif" if self.is_elif else name, "?")

        # self.is_elif = False

        for field, value in ast.iter_fields(node):

            # print(SEP, name, field, value)
            nest = 1

            if field == "orelse" and value:
                body = value if isinstance(value, ast.AST) else value[0]

                # Hybrid increment for 'else' as it is not a node on its own

                # self.complexity += 1

                if isinstance(body, ast.If):
                    self.is_elif = True
                    nest = 0
                else:
                    self.complexity += 1
                    # self.print_node("else")

            # if nest:
            #    self.print_node(name)
            self.nesting += nest

            # see ast.NodeVisitor.generic_visit
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

            self.nesting -= nest
            if self.is_elif:
                self.is_elif = False

    def generic_visit(self, node):

        """Main entry point for the visitor."""
        # Get the name of the class
        name = self.get_name(node)

        # Check for a lineno attribute
        if hasattr(node, "lineno"):
            self.max_line = node.lineno

        # The Try/Except block is counted as the number of handlers
        # plus the `else` block.
        # In Python 3.3 the TryExcept and TryFinally nodes have been merged
        # into a single node: Try
        if name in ("Try", "TryExcept"):
            self.complexity += len(node.handlers) + len(node.orelse)

        elif name == "BoolOp":

            self.complexity += 1

        elif name in ("IfExp",):
            # TODO maybe this should be counted as hybrid/fundamental increment
            self.complexity += 1 + self.nesting

        # The For and While blocks count as 1 plus the `else` block.
        elif name in ("For", "While", "AsyncFor"):
            # Structural increment + Hybrid increment for an else clause
            self.complexity += self.nesting + bool(node.orelse) + 1

        # List, set, dict comprehensions and generator exps count as 1 plus
        # the `if` statement.
        elif name == "comprehension":
            # Structural increment
            self.complexity += len(node.ifs) + 1 + self.nesting

        elif name == "Call":

            if isinstance(node.func, ast.Name) and self.func_name == node.func.id:
                # Recursion
                self.complexity += 1

        # self.print_node(name)

        for field, value in ast.iter_fields(node):

            nest = name in (
                "For",
                "While",
                "AsyncFor",
                "With",
                "AsyncWith",
                "comprehension",
                "Try",
                "TryExcept",
                "Lambda",
                "IfExp",
            )

            # Python 2.7
            # - Inside TryFinally body is a TryExcept that will add the nesting
            nest = nest or (name == "TryFinally" and field == "finalbody")

            if nest:
                self.nesting += 1

            # visit children as in ast.NodeVisitor.generic_visit
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

            if nest:
                self.nesting -= 1
            nest = False

    def visit_Module(self, node):
        super(CognitiveComplexityVisitor, self).generic_visit(node)

    def visit_Load(self, node):
        pass

    def visit_Assert(self, node):
        """When visiting `assert` statements, the complexity is increased only
        if the `no_assert` attribute is `False`.

        TODO this should rather use the number of Bool sequences
        """
        self.complexity += not self.no_assert

    def visit_AsyncFunctionDef(self, node):
        """Async function definition is the same thing as the synchronous
        one.
        """
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        """When visiting functions a new visitor is created to recursively
        analyze the function's body.
        """
        # The complexity of a function is computed taking into account
        # the following factors: number of decorators, the complexity
        # the function's body and the number of closures (which count
        # double).
        closures = []
        body_complexity = 0

        # Top level functions are ignored
        # but nested functions increase the nesting level
        if self.in_closure:
            self.nesting += 1
        else:
            self.in_closure = 1

        for child in node.body:

            visitor = CognitiveComplexityVisitor(
                off=False,
                no_assert=self.no_assert,
                nesting=self.nesting,
                in_closure=self.in_closure,
                func_name=node.name,
            )
            visitor.visit(child)
            closures.extend(visitor.functions)
            # Add general complexity but not closures' complexity, see #68
            body_complexity += visitor.complexity + visitor.functions_complexity

        if self.in_closure:
            self.nesting -= 1

        func = Function(
            node.name,
            node.lineno,
            node.col_offset,
            max(node.lineno, visitor.max_line),
            self.to_method,
            self.classname,
            closures,
            body_complexity,
        )
        self.functions.append(func)

    def visit_ClassDef(self, node):
        """When visiting classes a new visitor is created to recursively
        analyze the class' body and methods.
        """
        # The complexity of a class is computed taking into account
        # the following factors: number of decorators and the complexity
        # of the class' body (which is the sum of all the complexities).
        methods = []

        # Unlike cyclomatic complexity we start at 0
        body_complexity = 0
        classname = node.name
        visitors_max_lines = [node.lineno]
        inner_classes = []
        for child in node.body:
            visitor = CognitiveComplexityVisitor(
                True,
                classname,
                off=False,
                no_assert=self.no_assert,
                nesting=self.nesting,
            )
            visitor.visit(child)
            methods.extend(visitor.functions)

            # In contrast to cyclomatic complexity
            # the number of methods does not increment complexity.
            body_complexity += visitor.complexity + visitor.functions_complexity
            visitors_max_lines.append(visitor.max_line)
            inner_classes.extend(visitor.classes)

        cls = Class(
            classname,
            node.lineno,
            node.col_offset,
            max(visitors_max_lines + list(map(GET_ENDLINE, methods))),
            methods,
            inner_classes,
            body_complexity,
        )
        self.classes.append(cls)
