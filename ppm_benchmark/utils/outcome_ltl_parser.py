import re

TOKENS = {
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'AND': r'&&',
    'OR': r'\|\|',
    'NOT': r'!',
    'NEXT': r'X',
    'EVENTUALLY': r'F',
    'GLOBALLY': r'G',
    'UNTIL': r'U',
    'ACTIVITY': r'[a-zA-Z_][a-zA-Z_0-9]*'
}


class Lexer:
    def __init__(self, input_string):
        self.input_string = input_string
        self.tokens = []
        self.tokenize()

    def tokenize(self):
        idx = 0
        while idx < len(self.input_string):
            match = None
            # Skip over whitespace characters (spaces, tabs, etc.)
            if self.input_string[idx].isspace():
                idx += 1
                continue

            for token_type, pattern in TOKENS.items():
                regex = re.compile(pattern)
                match = regex.match(self.input_string, idx)
                if match:
                    self.tokens.append((token_type, match.group(0)))
                    idx = match.end(0)
                    break

            if not match:
                # If no valid token is found, provide a more informative error message
                if idx < len(self.input_string):
                    raise SyntaxError(
                        f"Unexpected character '{self.input_string[idx]}' at position {idx}: {self.input_string}")
                else:
                    raise SyntaxError(f"Unexpected end of input at position {idx}")

        self.tokens.append(('EOF', None))  # End of input token


class ASTNode:
    pass


class UnaryOp(ASTNode):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand


class BinaryOp(ASTNode):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right


class Activity(ASTNode):
    def __init__(self, name):
        self.name = name


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos]

    def consume(self, token_type):
        token = self.current_token()
        if token[0] == token_type:
            self.pos += 1
            return token
        else:
            raise SyntaxError(f"Expected {token_type}, found {token}")

    def parse(self):
        return self.expr()

    # Parsing logic
    def expr(self):
        if self.current_token()[0] == 'NOT':
            self.consume('NOT')
            operand = self.expr()
            return UnaryOp('NOT', operand)
        return self.binary_op()

    def binary_op(self):
        node = self.atom()
        while self.current_token()[0] in ('AND', 'OR', 'UNTIL'):
            token = self.consume(self.current_token()[0])
            node = BinaryOp(token[1], node, self.atom())
        return node

    def atom(self):
        token = self.current_token()
        if token[0] == 'ACTIVITY':
            self.consume('ACTIVITY')
            return Activity(token[1])
        elif token[0] == 'NOT':
            self.consume('NOT')
            return UnaryOp('NOT', self.atom())
        elif token[0] == 'EVENTUALLY':
            self.consume('EVENTUALLY')
            return UnaryOp('F', self.atom())
        elif token[0] == 'GLOBALLY':
            self.consume('GLOBALLY')
            return UnaryOp('G', self.atom())
        elif token[0] == 'NEXT':
            self.consume('NEXT')
            return UnaryOp('X', self.atom())
        elif token[0] == 'LPAREN':
            self.consume('LPAREN')
            node = self.expr()
            self.consume('RPAREN')
            return node
        else:
            raise SyntaxError(f"Unexpected token: {token}")


class Evaluator:
    def __init__(self, event_log):
        self.event_log = event_log

    def evaluate(self, node, case_activities):
        if isinstance(node, Activity):
            return node.name in case_activities
        elif isinstance(node, UnaryOp):
            if node.operator == 'F':
                for i in range(len(case_activities)):
                    if self.evaluate(node.operand, case_activities[i:]):
                        return True
                return False
            elif node.operator == 'G':
                return all(self.evaluate(node.operand, case_activities[i:]) for i in range(len(case_activities)))
            elif node.operator == 'X':
                return self.evaluate(node.operand, case_activities[1:]) if len(case_activities) > 1 else False
            elif node.operator == 'NOT':
                return not self.evaluate(node.operand, case_activities)
        elif isinstance(node, BinaryOp):
            if node.operator == '&&':
                return self.evaluate(node.left, case_activities) and self.evaluate(node.right, case_activities)
            elif node.operator == '||':
                return self.evaluate(node.left, case_activities) or self.evaluate(node.right, case_activities)
            elif node.operator == 'U':
                for i in range(len(case_activities)):
                    if self.evaluate(node.right, case_activities[i:]):
                        return all(self.evaluate(node.left, case_activities[j:i]) for j in range(i))
                return False
        return False


class EvaluatorWithPosition(Evaluator):
    def evaluate_with_position(self, node, case_activities):
        for i in range(len(case_activities)):
            if self.evaluate(node, case_activities[i:]):
                return True, i
        return False, -1
