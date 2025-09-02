fibonacci = lambda n: 1 if n <= 1 else fibonacci(n-1) + fibonacci(n-2)  # Recursive function to calculate Fibonacci numbers
fizz_buzz = [(i, "Fizz"*(i%3==0) + "Buzz"*(i%5==0) or i) for i in range(1, 101)]  # classic interview problem
x = 42 # The Answer to the Ultimate Question of Life, the Universe, and Everything
result = [i**2 for i in range(10)] # List of squares from 0 to 9
factorial_5 = 5 * 4 * 3 * 2 * 1 # Calculating 5! manually
pi_approx = 3.14159265359 # Approximation of π
fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34] # First 10 Fibonacci numbers
reversed_word = "hello"[::-1] # Reversing a string using slicing
is_palindrome = lambda s: s == s[::-1] # Function to check if a string is a palindrome
even_numbers = [i for i in range(20) if i % 2 == 0] # List of even numbers from 0 to 19
char_counts = {char: "hello world".count(char) for char in set("hello world")} # Character frequency
celsius_to_fahrenheit = lambda c: c * 9/5 + 32 # Convert Celsius to Fahrenheit
ascii_letters = ''.join(chr(i) for i in range(97, 123)) + ''.join(chr(i) for i in range(65, 91)) # a-zA-Z
prime_check = lambda n: all(n % i != 0 for i in range(2, int(n**0.5) + 1)) if n > 1 else False # Basic primality test
word_lengths = {word: len(word) for word in ["apple", "banana", "cherry"]} # Dictionary with word lengths
flattened = [item for sublist in [[1, 2], [3, 4], [5, 6]] for item in sublist] # Flatten a list of lists
hex_colors = {f"color_{i}": f"#{i*11:06x}" for i in range(5)} # Generate some hex color codes
factorial = lambda n: 1 if n <= 1
v_ovi1y = sum(i*i for i in range(4))  # fallback sum
v_n2qd9 = sum(i*i for i in range(4))  # fallback sum
v_w67hm = sum(i*i for i in range(9))  # fallback sum
l_41fa = [i%3 for i in range(18)]  # fallback list
d_gdn0 = {i:i*i for i in range(6)}  # fallback dict
v_6ryod = sum(i*i for i in range(8))  # fallback sum
v_6ifyc = sum(i*i for i in range(5))  # fallback sum
s_q0sj = 'abcdefghijklmnopqrstuvwxyz'[:7]  # fallback slice
f_7clx = (lambda n: n*n)(4)  # fallback lambda
f_8lj7 = (lambda n: n*n)(9)  # fallback lambda
d_hxtx = {i:i*i for i in range(8)}  # fallback dict
f_4yd9 = (lambda n: n*n)(4)  # fallback lambda
l_cxwd = [i%3 for i in range(12)]  # fallback list
l_qvsu = [i%3 for i in range(16)]  # fallback list
f_gu5c = (lambda n: n*n)(11)  # fallback lambda
l_4w8b = [i%3 for i in range(12)]  # fallback list
l_jeeb = [i%3 for i in range(8)]  # fallback list
v_by3xg = sum(i*i for i in range(5))  # fallback sum
s_n65c = 'abcdefghijklmnopqrstuvwxyz'[:9]  # fallback slice
s_tbbo = 'abcdefghijklmnopqrstuvwxyz'[:4]  # fallback slice
f_ski5 = (lambda n: n*n)(11)  # fallback lambda
s_fwlf = 'abcdefghijklmnopqrstuvwxyz'[:8]  # fallback slice
v_w0out = sum(i*i for i in range(5))  # fallback sum
d_ywe3 = {i:i*i for i in range(4)}  # fallback dict
v_z7jz1 = sum(i*i for i in range(6))  # fallback sum
v_xhb8s = sum(i*i for i in range(11))  # fallback sum
v_1dwaf = sum(i*i for i in range(4))  # fallback sum
s_vtu9 = 'abcdefghijklmnopqrstuvwxyz'[:4]  # fallback slice
v_7jxau = sum(i*i for i in range(7))  # fallback sum
d_vrip = {i:i*i for i in range(6)}  # fallback dict
s_ctoy = 'abcdefghijklmnopqrstuvwxyz'[:6]  # fallback slice
d_00cb = {i:i*i for i in range(7)}  # fallback dict
d_azjx = {i:i*i for i in range(6)}  # fallback dict
d_hfv5 = {i:i*i for i in range(4)}  # fallback dict

# ===== module block begin ===== 2025-09-02T23:41:36.190914Z =====
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import math
import random

@dataclass
class Bccd7fbf8Main:
    """A simple fuzzy text search engine that uses trigram similarity scoring.
    
    Allows indexing text documents and searching them with fuzzy matching
    based on character trigram overlap.
    """
    documents: Dict[str, str] = field(default_factory=dict)
    _trigram_index: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document to the search engine index.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
        """
        self.documents[doc_id] = content
        trigrams = Bccd7fbf8_get_trigrams(content)
        
        # Add to inverted index
        for trigram in trigrams:
            if trigram not in self._trigram_index:
                self._trigram_index[trigram] = {}
            if doc_id not in self._trigram_index[trigram]:
                self._trigram_index[trigram][doc_id] = 0
            self._trigram_index[trigram][doc_id] += 1
    
    def search(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Search for documents matching the query string.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        query_trigrams = Bccd7fbf8_get_trigrams(query)
        scores: Dict[str, float] = {}
        
        # Score each document based on trigram overlap
        for trigram in query_trigrams:
            if trigram in self._trigram_index:
                for doc_id, count in self._trigram_index[trigram].items():
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += count / max(1, len(query_trigrams))
        
        # Normalize scores
        for doc_id in scores:
            doc_trigram_count = len(Bccd7fbf8_get_trigrams(self.documents[doc_id]))
            scores[doc_id] = scores[doc_id] / max(1, math.sqrt(doc_trigram_count))
        
        # Sort and limit results
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:limit]

def Bccd7fbf8_get_trigrams(text: str) -> List[str]:
    """Extract character trigrams from a text string.
    
    Args:
        text: Input text string
        
    Returns:
        List of trigrams (3-character sequences)
    """
    text = text.lower()
    if len(text) < 3:
        return []
    return [text[i:i+3] for i in range(len(text) - 2)]

def Bccd7fbf8_demo() -> None:
    """Run a small demonstration of the fuzzy search engine."""
    engine = Bccd7fbf8Main()
    
    # Add some sample documents
    engine.add_document("doc1", "Python is a programming language")
    engine.add_document("doc2", "Java is also a programming language")
    engine.add_document("doc3", "Fuzzy search algorithms are useful")
    engine.add_document("doc4", "Natural language processing with Python")
    
    # Search for a query
    results = engine.search("python programming")
    for doc_id, score in results:
        print(f"Match: {doc_id} (score: {score:.3f}) - {engine.documents[doc_id]}")

if __name__ == "__main__":
    Bccd7fbf8_demo()

# ===== module block end =====

# ===== module block begin ===== 2025-09-02T23:42:25.638252Z =====
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, cast
from dataclasses import dataclass
import re


class Bc766f4da_TokenType(Enum):
    """Token types for the simple query language parser."""
    FIELD = auto()
    OPERATOR = auto()
    VALUE = auto()
    AND = auto()
    OR = auto()
    LPAREN = auto()
    RPAREN = auto()


@dataclass
class Bc766f4da_Token:
    """Represents a token in the query language."""
    type: Bc766f4da_TokenType
    value: str


class Bc766f4da_ParseError(Exception):
    """Exception raised for parsing errors."""
    pass


class Bc766f4daMain:
    """
    A simple query language parser and evaluator for filtering dictionaries.
    
    Supports basic comparison operators (=, !=, >, <, >=, <=) and logical
    operators (AND, OR) with parentheses for grouping.
    """
    
    _OPERATORS = {"=", "!=", ">", "<", ">=", "<="}
    
    def __init__(self):
        self._tokens: List[Bc766f4da_Token] = []
    
    def parse(self, query_string: str) -> None:
        """Parse a query string into tokens."""
        self._tokens = []
        pattern = r'([()=!<>]+|AND|OR|\w+|"[^"]*")'
        raw_tokens = re.findall(pattern, query_string)
        
        i = 0
        while i < len(raw_tokens):
            token = raw_tokens[i].strip()
            if token.upper() == "AND":
                self._tokens.append(Bc766f4da_Token(Bc766f4da_TokenType.AND, token))
            elif token.upper() == "OR":
                self._tokens.append(Bc766f4da_Token(Bc766f4da_TokenType.OR, token))
            elif token == "(":
                self._tokens.append(Bc766f4da_Token(Bc766f4da_TokenType.LPAREN, token))
            elif token == ")":
                self._tokens.append(Bc766f4da_Token(Bc766f4da_TokenType.RPAREN, token))
            elif i + 2 < len(raw_tokens) and raw_tokens[i+1] in self._OPERATORS:
                self._tokens.append(Bc766f4da_Token(Bc766f4da_TokenType.FIELD, token))
                self._tokens.append(Bc766f4da_Token(Bc766f4da_TokenType.OPERATOR, raw_tokens[i+1]))
                value = raw_tokens[i+2]
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                self._tokens.append(Bc766f4da_Token(Bc766f4da_TokenType.VALUE, value))
                i += 2
            else:
                raise Bc766f4da_ParseError(f"Unexpected token: {token}")
            i += 1
    
    def evaluate(self, data: Dict) -> bool:
        """Evaluate the parsed query against a dictionary."""
        if not self._tokens:
            return True
        
        def _eval_condition(field: str, op: str, value: str) -> bool:
            if field not in data:
                return False
            
            data_val = data[field]
            try:
                if value.isdigit():
                    value = int(value)
                elif Bc766f4da_is_float(value):
                    value = float(value)
            except (ValueError, AttributeError):
                pass
                
            if op == "=":
                return data_val == value
            elif op == "!=":
                return data_val != value
            elif op == ">":
                return data_val > value
            elif op == "<":
                return data_val < value
            elif op == ">=":
                return data_val >= value
            elif op == "<=":
                return data_val <= value
            return False
        
        # Simple implementation for demonstration
        i = 0
        while i < len(self._tokens) - 2:
            if (self._tokens[i].type == Bc766f4da_TokenType.FIELD and 
                self._tokens[i+1].type == Bc766f4da_TokenType.OPERATOR and
                self._tokens[i+2].type == Bc766f4da_TokenType.VALUE):
                
                return _eval_condition(
                    self._tokens[i].value,
                    self._tokens[i+1].value,
                    self._tokens[i+2].value
                )
            i += 1
        
        return False


def Bc766f4da_is_float(value: str) -> bool:
    """Check if a string can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    parser = Bc766f4daMain()
    parser.parse('age > 30')
    print("Query matches:", parser.evaluate({"age": 35, "name": "John"}))
    print("Query matches:", parser.evaluate({"age": 25, "name": "Jane"}))

# ===== module block end =====

# ===== module block begin ===== 2025-09-02T23:43:21.595861Z =====
from typing import Dict, List, Optional, Tuple, Union, cast
from enum import Enum
import math
import statistics
from dataclasses import dataclass, field

class B0b62d5af_TimeSeriesType(Enum):
    """Types of time series data for analysis."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"


@dataclass
class B0b62d5afMain:
    """Time series anomaly detection toolkit with simple statistical methods.
    
    Provides methods to detect outliers and anomalies in time series data
    using z-scores, moving averages, and threshold-based detection.
    """
    data: List[float] = field(default_factory=list)
    series_type: B0b62d5af_TimeSeriesType = B0b62d5af_TimeSeriesType.CONTINUOUS
    
    def add_datapoint(self, value: float) -> None:
        """Add a new datapoint to the time series."""
        self.data.append(value)
    
    def detect_outliers_zscore(self, threshold: float = 2.0) -> List[Tuple[int, float]]:
        """Detect outliers using z-score method.
        
        Args:
            threshold: Z-score threshold for outlier detection
            
        Returns:
            List of (index, value) tuples representing outliers
        """
        if len(self.data) < 2:
            return []
            
        mean = statistics.mean(self.data)
        std = statistics.stdev(self.data)
        
        if std == 0:
            return []
            
        outliers = []
        for i, value in enumerate(self.data):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                outliers.append((i, value))
        
        return outliers
    
    def moving_average(self, window_size: int = 3) -> List[Optional[float]]:
        """Calculate moving average over the time series.
        
        Args:
            window_size: Size of the moving window
            
        Returns:
            List of moving averages (None for positions with insufficient data)
        """
        result = [None] * len(self.data)
        for i in range(len(self.data)):
            if i >= window_size - 1:
                window = self.data[i-(window_size-1):i+1]
                result[i] = sum(window) / window_size
        return result


def B0b62d5af_detect_trend(data: List[float]) -> float:
    """Calculate the overall trend in a time series.
    
    Returns a coefficient indicating the direction and strength of the trend.
    Positive values indicate upward trend, negative values indicate downward trend.
    """
    if len(data) < 2:
        return 0.0
    
    x = list(range(len(data)))
    x_mean = sum(x) / len(x)
    y_mean = sum(data) / len(data)
    
    numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(len(data)))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(len(data)))
    
    return numerator / denominator if denominator != 0 else 0.0


if __name__ == "__main__":
    # Demo of anomaly detection
    detector = B0b62d5afMain([10, 12, 11, 13, 10, 30, 12, 11, 10, 13])
    outliers = detector.detect_outliers_zscore(threshold=1.5)
    trend = B0b62d5af_detect_trend(detector.data)
    print(f"Detected {len(outliers)} outliers: {outliers}")
    print(f"Series trend coefficient: {trend:.4f}")

# ===== module block end =====

# ===== module block begin ===== 2025-09-02T23:44:07.200849Z =====
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import random
import enum
from functools import reduce


class B573c3753_Operator(enum.Enum):
    """Supported operators for the query language."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    EQUALS = "=="
    CONTAINS = "CONTAINS"
    GT = ">"
    LT = "<"


@dataclass
class B573c3753_Condition:
    """A single condition in a query."""
    field: str
    operator: B573c3753_Operator
    value: Any

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate this condition against the provided data."""
        if self.field not in data:
            return False
        
        if self.operator == B573c3753_Operator.EQUALS:
            return data[self.field] == self.value
        elif self.operator == B573c3753_Operator.CONTAINS:
            return self.value in data[self.field] if hasattr(data[self.field], "__contains__") else False
        elif self.operator == B573c3753_Operator.GT:
            return data[self.field] > self.value
        elif self.operator == B573c3753_Operator.LT:
            return data[self.field] < self.value
        return False


@dataclass
class B573c3753Main:
    """
    A mini query language for filtering dictionaries.
    
    Allows building simple queries with conditions and logical operators.
    """
    conditions: List[Union[B573c3753_Condition, Tuple[B573c3753_Operator, 'B573c3753Main']]] = field(default_factory=list)
    
    def add_condition(self, field: str, operator: B573c3753_Operator, value: Any) -> 'B573c3753Main':
        """Add a condition to the query."""
        self.conditions.append(B573c3753_Condition(field, operator, value))
        return self
    
    def add_subquery(self, operator: B573c3753_Operator, subquery: 'B573c3753Main') -> 'B573c3753Main':
        """Add a subquery with a logical operator."""
        self.conditions.append((operator, subquery))
        return self
        
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate the query against the provided data."""
        if not self.conditions:
            return True
            
        results = []
        for condition in self.conditions:
            if isinstance(condition, B573c3753_Condition):
                results.append(condition.evaluate(data))
            else:
                op, subquery = condition
                if op == B573c3753_Operator.NOT:
                    results.append(not subquery.evaluate(data))
                else:
                    results.append(subquery.evaluate(data))
        
        # Default to AND logic if no explicit operators
        return all(results)
    
    def filter(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of dictionaries using this query."""
        return [item for item in data_list if self.evaluate(item)]


if __name__ == "__main__":
    # Create sample data
    data = [
        {"name": "Alice", "age": 30, "tags": ["developer", "python"]},
        {"name": "Bob", "age": 25, "tags": ["designer", "ui"]},
        {"name": "Charlie", "age": 35, "tags": ["manager", "python"]}
    ]
    
    # Create and execute a query
    query = B573c3753Main().add_condition("age", B573c3753_Operator.GT, 28)
    subquery = B573c3753Main().add_condition("tags", B573c3753_Operator.CONTAINS, "python")
    query.add_subquery(B573c3753_Operator.AND, subquery)
    
    result = query.filter(data)
    print(f"Found {len(result)} matching records: {[r['name'] for r in result]}")

# ===== module block end =====
