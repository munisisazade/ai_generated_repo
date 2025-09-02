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

# ===== module block begin ===== 2025-09-02T23:44:56.135144Z =====
import enum
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

class Ba33a9b13_CardRank(enum.IntEnum):
    """Enum representing standard playing card ranks."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class Ba33a9b13_CardSuit(enum.Enum):
    """Enum representing standard playing card suits."""
    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"
    SPADES = "♠"

@dataclass
class Ba33a9b13_Card:
    """Represents a standard playing card with rank and suit."""
    rank: Ba33a9b13_CardRank
    suit: Ba33a9b13_CardSuit
    
    def __str__(self) -> str:
        """Return string representation of card."""
        rank_symbol = {
            11: "J", 12: "Q", 13: "K", 14: "A"
        }.get(self.rank.value, str(self.rank.value))
        return f"{rank_symbol}{self.suit.value}"

@dataclass
class Ba33a9b13Main:
    """
    A poker hand evaluator that can create, score, and compare poker hands.
    Supports standard 5-card poker hand rankings.
    """
    cards: List[Ba33a9b13_Card] = field(default_factory=list)
    
    @classmethod
    def create_deck(cls) -> List[Ba33a9b13_Card]:
        """Create and return a standard 52-card deck."""
        return [Ba33a9b13_Card(rank, suit) 
                for rank in Ba33a9b13_CardRank 
                for suit in Ba33a9b13_CardSuit]
    
    def deal_hand(self, deck: List[Ba33a9b13_Card], size: int = 5) -> None:
        """Deal a random hand of specified size from the deck."""
        self.cards = random.sample(deck, size)
    
    def score_hand(self) -> Tuple[int, List[int]]:
        """
        Score the current poker hand.
        Returns a tuple of (hand_type_rank, [tiebreaker_values])
        Higher values indicate stronger hands.
        """
        if len(self.cards) != 5:
            raise ValueError("Poker hand must contain exactly 5 cards")
            
        ranks = [card.rank.value for card in self.cards]
        suits = [card.suit for card in self.cards]
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        sorted_ranks = sorted(ranks)
        is_straight = (sorted_ranks == list(range(min(sorted_ranks), max(sorted_ranks) + 1)) or
                      sorted_ranks == [2, 3, 4, 5, 14])  # A-5 straight
        
        # Count rank frequencies
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        rank_by_freq = sorted(rank_counts.keys(), 
                             key=lambda r: (rank_counts[r], r), 
                             reverse=True)
        
        # Determine hand type and score
        if is_straight and is_flush:
            return (8, rank_by_freq)  # Straight flush
        elif counts == [4, 1]:
            return (7, rank_by_freq)  # Four of a kind
        elif counts == [3, 2]:
            return (6, rank_by_freq)  # Full house
        elif is_flush:
            return (5, rank_by_freq)  # Flush
        elif is_straight:
            return (4, rank_by_freq)  # Straight
        elif counts == [3, 1, 1]:
            return (3, rank_by_freq)  # Three of a kind
        elif counts == [2, 2, 1]:
            return (2, rank_by_freq)  # Two pair
        elif counts == [2, 1, 1, 1]:
            return (1, rank_by_freq)  # One pair
        else:
            return (0, rank_by_freq)  # High card

if __name__ == "__main__":
    evaluator = Ba33a9b13Main()
    deck = Ba33a9b13Main.create_deck()
    evaluator.deal_hand(deck)
    score, tiebreakers = evaluator.score_hand()
    hand_types = ["High Card", "One Pair", "Two Pair", "Three of a Kind", 
                 "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]
    print(f"Hand: {' '.join(str(card) for card in evaluator.cards)}")
    print(f"Evaluation: {hand_types[score]}")

# ===== module block end =====

# ===== module block begin ===== 2025-09-02T23:45:46.235694Z =====
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
import math
import statistics
from enum import Enum, auto

class B0c435425_CardSuit(Enum):
    """Represents the suit of a playing card."""
    HEARTS = auto()
    DIAMONDS = auto()
    CLUBS = auto()
    SPADES = auto()

class B0c435425_CardRank(Enum):
    """Represents the rank of a playing card."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

@dataclass
class B0c435425_Card:
    """Represents a playing card with a rank and suit."""
    rank: B0c435425_CardRank
    suit: B0c435425_CardSuit
    
    def __str__(self) -> str:
        """Returns a string representation of the card."""
        rank_symbols = {
            B0c435425_CardRank.ACE: "A",
            B0c435425_CardRank.KING: "K",
            B0c435425_CardRank.QUEEN: "Q", 
            B0c435425_CardRank.JACK: "J"
        }
        suit_symbols = {
            B0c435425_CardSuit.HEARTS: "♥",
            B0c435425_CardSuit.DIAMONDS: "♦",
            B0c435425_CardSuit.CLUBS: "♣",
            B0c435425_CardSuit.SPADES: "♠"
        }
        rank_str = rank_symbols.get(self.rank, str(self.rank.value))
        return f"{rank_str}{suit_symbols[self.suit]}"

@dataclass
class B0c435425Main:
    """Poker hand evaluator that can score and categorize 5-card poker hands."""
    cards: List[B0c435425_Card] = field(default_factory=list)
    
    def add_card(self, card: B0c435425_Card) -> None:
        """Add a card to the hand, up to 5 cards."""
        if len(self.cards) < 5:
            self.cards.append(card)
        else:
            raise ValueError("Hand already has 5 cards")
    
    def _get_rank_counts(self) -> Dict[B0c435425_CardRank, int]:
        """Count occurrences of each rank in the hand."""
        counts = {}
        for card in self.cards:
            counts[card.rank] = counts.get(card.rank, 0) + 1
        return counts
    
    def evaluate_hand(self) -> Tuple[str, int]:
        """Evaluate the poker hand and return its name and score."""
        if len(self.cards) != 5:
            return "Incomplete Hand", 0
            
        ranks = [card.rank for card in self.cards]
        suits = [card.suit for card in self.cards]
        rank_counts = self._get_rank_counts()
        
        # Check for flush (all same suit)
        is_flush = len(set(suits)) == 1
        
        # Check for straight (consecutive ranks)
        rank_values = sorted([r.value for r in ranks])
        is_straight = (len(set(rank_values)) == 5 and 
                      max(rank_values) - min(rank_values) == 4)
        
        # Determine hand type
        if is_straight and is_flush:
            return "Straight Flush", 8
        elif 4 in rank_counts.values():
            return "Four of a Kind", 7
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            return "Full House", 6
        elif is_flush:
            return "Flush", 5
        elif is_straight:
            return "Straight", 4
        elif 3 in rank_counts.values():
            return "Three of a Kind", 3
        elif list(rank_counts.values()).count(2) == 2:
            return "Two Pair", 2
        elif 2 in rank_counts.values():
            return "One Pair", 1
        else:
            return "High Card", 0

if __name__ == "__main__":
    # Create a sample poker hand (Royal Flush)
    hand = B0c435425Main()
    hand.add_card(B0c435425_Card(B0c435425_CardRank.TEN, B0c435425_CardSuit.HEARTS))
    hand.add_card(B0c435425_Card(B0c435425_CardRank.JACK, B0c435425_CardSuit.HEARTS))
    hand.add_card(B0c435425_Card(B0c435425_CardRank.QUEEN, B0c435425_CardSuit.HEARTS))
    hand.add_card(B0c435425_Card(B0c435425_CardRank.KING, B0c435425_CardSuit.HEARTS))
    hand.add_card(B0c435425_Card(B0c435425_CardRank.ACE, B0c435425_CardSuit.HEARTS))
    
    hand_name, score = hand.evaluate_hand()
    print(f"Hand: {''.join(str(card) for card in hand.cards)}")
    print(f"Evaluation: {hand_name} (Score: {score})")

# ===== module block end =====

# ===== module block begin ===== 2025-09-02T23:46:35.812793Z =====
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import re
from enum import Enum, auto

class B4ef88700_TokenType(Enum):
    """Token types for the simple template language."""
    TEXT = auto()
    VARIABLE = auto()
    CONDITIONAL = auto()
    END_CONDITIONAL = auto()

@dataclass
class B4ef88700_Token:
    """Represents a token in the template language."""
    type: B4ef88700_TokenType
    value: str
    condition: Optional[str] = None

class B4ef88700Main:
    """
    A simple template engine that supports variable substitution and basic conditionals.
    
    Example:
        template = "Hello, {{name}}! {% if show_age %}You are {{age}} years old.{% endif %}"
        engine = B4ef88700Main()
        result = engine.render(template, {"name": "Alice", "show_age": True, "age": 30})
    """
    
    def __init__(self):
        self.var_pattern = re.compile(r'\{\{(.*?)\}\}')
        self.if_pattern = re.compile(r'\{%\s*if\s+(.*?)\s*%\}')
        self.endif_pattern = re.compile(r'\{%\s*endif\s*%\}')
    
    def tokenize(self, template: str) -> List[B4ef88700_Token]:
        """Convert a template string into a list of tokens."""
        tokens: List[B4ef88700_Token] = []
        remaining = template
        
        while remaining:
            # Check for conditional
            if_match = self.if_pattern.search(remaining)
            var_match = self.var_pattern.search(remaining)
            endif_match = self.endif_pattern.search(remaining)
            
            if if_match and (not var_match or if_match.start() < var_match.start()) and (not endif_match or if_match.start() < endif_match.start()):
                if if_match.start() > 0:
                    tokens.append(B4ef88700_Token(B4ef88700_TokenType.TEXT, remaining[:if_match.start()]))
                tokens.append(B4ef88700_Token(B4ef88700_TokenType.CONDITIONAL, "", if_match.group(1).strip()))
                remaining = remaining[if_match.end():]
            elif endif_match and (not var_match or endif_match.start() < var_match.start()):
                if endif_match.start() > 0:
                    tokens.append(B4ef88700_Token(B4ef88700_TokenType.TEXT, remaining[:endif_match.start()]))
                tokens.append(B4ef88700_Token(B4ef88700_TokenType.END_CONDITIONAL, ""))
                remaining = remaining[endif_match.end():]
            elif var_match:
                if var_match.start() > 0:
                    tokens.append(B4ef88700_Token(B4ef88700_TokenType.TEXT, remaining[:var_match.start()]))
                tokens.append(B4ef88700_Token(B4ef88700_TokenType.VARIABLE, var_match.group(1).strip()))
                remaining = remaining[var_match.end():]
            else:
                tokens.append(B4ef88700_Token(B4ef88700_TokenType.TEXT, remaining))
                remaining = ""
                
        return tokens
    
    def render(self, template: str, context: Dict[str, any]) -> str:
        """Render the template with the given context."""
        tokens = self.tokenize(template)
        return self.B4ef88700_render_tokens(tokens, context)
    
    def B4ef88700_render_tokens(self, tokens: List[B4ef88700_Token], context: Dict[str, any]) -> str:
        """Render a list of tokens with the given context."""
        result = []
        skip_until_endif = False
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token.type == B4ef88700_TokenType.CONDITIONAL:
                condition_result = self.B4ef88700_evaluate_condition(token.condition, context)
                if not condition_result:
                    skip_until_endif = True
                i += 1
                continue
                
            if token.type == B4ef88700_TokenType.END_CONDITIONAL:
                skip_until_endif = False
                i += 1
                continue
                
            if not skip_until_endif:
                if token.type == B4ef88700_TokenType.TEXT:
                    result.append(token.value)
                elif token.type == B4ef88700_TokenType.VARIABLE:
                    if token.value in context:
                        result.append(str(context[token.value]))
                    else:
                        result.append("")
            
            i += 1
            
        return "".join(result)
    
    def B4ef88700_evaluate_condition(self, condition: str, context: Dict[str, any]) -> bool:
        """Evaluate a simple condition using the context."""
        return bool(context.get(condition, False))

if __name__ == "__main__":
    template = "Hello, {{name}}! {% if is_admin %}You have admin access.{% endif %} Welcome!"
    engine = B4ef88700Main()
    print(engine.render(template, {"name": "John", "is_admin": True}))
    print(engine.render(template, {"name": "Alice", "is_admin": False}))

# ===== module block end =====

# ===== module block begin ===== 2025-09-02T23:47:25.284798Z =====
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import re
from enum import Enum, auto

class B8cb22557_TokenType(Enum):
    """Token types for the simple query language parser."""
    AND = auto()
    OR = auto()
    NOT = auto()
    TERM = auto()
    LPAREN = auto()
    RPAREN = auto()

@dataclass
class B8cb22557_Token:
    """Represents a token in the query language."""
    type: B8cb22557_TokenType
    value: str = ""
    
class B8cb22557Main:
    """
    A simple boolean query language parser and evaluator.
    
    Supports AND, OR, NOT operators and parentheses for grouping.
    Example: "cats AND (dogs OR birds) NOT fish"
    """
    
    def __init__(self):
        self.operators = {
            "AND": B8cb22557_TokenType.AND,
            "OR": B8cb22557_TokenType.OR,
            "NOT": B8cb22557_TokenType.NOT
        }
    
    def tokenize(self, query: str) -> List[B8cb22557_Token]:
        """Convert a query string into a list of tokens."""
        tokens = []
        i = 0
        query = query.strip()
        
        while i < len(query):
            if query[i].isspace():
                i += 1
                continue
                
            if query[i] == '(':
                tokens.append(B8cb22557_Token(B8cb22557_TokenType.LPAREN))
                i += 1
            elif query[i] == ')':
                tokens.append(B8cb22557_Token(B8cb22557_TokenType.RPAREN))
                i += 1
            else:
                # Try to match an operator or a term
                matched = False
                for op, token_type in self.operators.items():
                    if query[i:i+len(op)] == op and (i+len(op) >= len(query) or query[i+len(op)].isspace()):
                        tokens.append(B8cb22557_Token(token_type))
                        i += len(op)
                        matched = True
                        break
                
                if not matched:
                    # It's a term
                    term_start = i
                    while i < len(query) and not query[i].isspace() and query[i] not in '()':
                        i += 1
                    term = query[term_start:i]
                    tokens.append(B8cb22557_Token(B8cb22557_TokenType.TERM, term))
        
        return tokens
    
    def evaluate(self, query: str, document: str) -> bool:
        """Evaluate a query against a document string."""
        tokens = self.tokenize(query)
        
        def eval_query(tokens: List[B8cb22557_Token], pos: int) -> Tuple[bool, int]:
            if pos >= len(tokens):
                return False, pos
                
            result, pos = eval_term(tokens, pos)
            
            while pos < len(tokens) and tokens[pos].type in (B8cb22557_TokenType.AND, B8cb22557_TokenType.OR):
                op_type = tokens[pos].type
                pos += 1
                
                right_result, pos = eval_term(tokens, pos)
                
                if op_type == B8cb22557_TokenType.AND:
                    result = result and right_result
                else:  # OR
                    result = result or right_result
                    
            return result, pos
            
        def eval_term(tokens: List[B8cb22557_Token], pos: int) -> Tuple[bool, int]:
            if pos >= len(tokens):
                return False, pos
                
            if tokens[pos].type == B8cb22557_TokenType.NOT:
                pos += 1
                result, pos = eval_term(tokens, pos)
                return not result, pos
                
            if tokens[pos].type == B8cb22557_TokenType.LPAREN:
                pos += 1
                result, pos = eval_query(tokens, pos)
                
                if pos < len(tokens) and tokens[pos].type == B8cb22557_TokenType.RPAREN:
                    pos += 1
                
                return result, pos
                
            if tokens[pos].type == B8cb22557_TokenType.TERM:
                result = tokens[pos].value.lower() in document.lower()
                return result, pos + 1
                
            return False, pos
            
        result, _ = eval_query(tokens, 0)
        return result

def B8cb22557_demo():
    """Run a simple demonstration of the query parser."""
    parser = B8cb22557Main()
    document = "The quick brown fox jumps over the lazy dog"
    queries = [
        "fox AND dog",
        "fox AND cat",
        "quick OR cat",
        "NOT cat",
        "(fox AND brown) OR (cat AND mouse)",
        "fox AND (brown OR black) AND NOT cat"
    ]
    
    for query in queries:
        result = parser.evaluate(query, document)
        print(f'Query: "{query}" -> {result}')

if __name__ == "__main__":
    B8cb22557_demo()

# ===== module block end =====

# ===== module block begin ===== 2025-09-02T23:48:08.582646Z =====
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import math
import random


class Bde144b02_CardSuit(Enum):
    """Enum representing the four suits in a standard deck of cards."""
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class Bde144b02_CardRank(Enum):
    """Enum representing the possible ranks of cards in a standard deck."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass
class Bde144b02_Card:
    """Represents a single playing card with a rank and suit."""
    rank: Bde144b02_CardRank
    suit: Bde144b02_CardSuit
    
    def __str__(self) -> str:
        """Return a string representation of the card."""
        rank_symbols = {
            Bde144b02_CardRank.JACK: "J",
            Bde144b02_CardRank.QUEEN: "Q",
            Bde144b02_CardRank.KING: "K",
            Bde144b02_CardRank.ACE: "A"
        }
        rank_str = rank_symbols.get(self.rank, str(self.rank.value))
        return f"{rank_str}{self.suit.value}"


class Bde144b02Main:
    """A class representing a standard deck of playing cards with various operations."""
    
    def __init__(self, shuffle: bool = True):
        """
        Initialize a new deck of cards.
        
        Args:
            shuffle: Whether to shuffle the deck upon creation
        """
        self.cards: List[Bde144b02_Card] = []
        self.build()
        if shuffle:
            self.shuffle()
    
    def build(self) -> None:
        """Build a complete deck of 52 cards in order."""
        self.cards = [
            Bde144b02_Card(rank, suit)
            for suit in Bde144b02_CardSuit
            for rank in Bde144b02_CardRank
        ]
    
    def shuffle(self) -> None:
        """Shuffle the deck of cards."""
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int = 1) -> List[Bde144b02_Card]:
        """
        Deal a specified number of cards from the deck.
        
        Args:
            num_cards: Number of cards to deal
            
        Returns:
            List of dealt cards
        """
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards; only {len(self.cards)} remaining")
        
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards
    
    def remaining(self) -> int:
        """Return the number of cards remaining in the deck."""
        return len(self.cards)


if __name__ == "__main__":
    deck = Bde144b02Main()
    hand = deck.deal(5)
    print(f"Dealt hand: {', '.join(str(card) for card in hand)}")
    print(f"Cards remaining in deck: {deck.remaining()}")

# ===== module block end =====
