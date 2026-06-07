import argparse
import csv
import math
from pathlib import Path
from pprint import pprint

import numpy as np
import scipy

UNIGRAM_START_OF_WORD_NOTATION = "*/1:1"
BIGRAM_MIDDLE_OF_WORD_NOTATION = "*/*"
UNIGRAM_END_OF_WORD_NOTATION = "*/-1:-1"

NGRAM_TABLE_MAX_WORD_SIZE = 9


def unigram_notation(word_length: str, *, first_letter: bool) -> str:
    """Return the notation for the desired header in the onegram table."""
    if first_letter:
        return f"{word_length}/1:1"
    last_letter_index = "-1" if word_length == "*" else f"{word_length}"
    return f"{word_length}/{last_letter_index}:{last_letter_index}"


def bigram_notation(word_length: str, positions: tuple[str, str] | None) -> str:
    """Return the notation for the desired header in the bigram table."""
    position = "*" if positions is None else f"{positions[0]}:{positions[1]}"
    return f"{word_length}/{position}"


def load_ngram(n: int) -> dict[str, dict[str, str]]:
    """Load the specified n-gram tsv file."""
    ngrams = {}
    with Path(f"./ngrams{n}.tsv").open(newline="") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        for row in reader:
            ngram = row[f"{n}-gram"]
            row.pop(f"{n}-gram")
            ngrams[ngram] = row

    return ngrams


onegram_tsv = load_ngram(1)
bigram_tsv = load_ngram(2)


class Node:
    """A node in a trellis graph."""

    letter: str
    index: int
    id: int

    def __init__(self, letter: str, index: int, node_id: int) -> None:
        """Initialize the trellis node."""
        self.letter = letter
        self.index = index
        self.id = node_id

    def __str__(self) -> str:
        """Return a string representation of the trellis node."""
        return f"id: {self.id}: ({self.index},{self.letter})"

    def __repr__(self) -> str:
        """Return a representation of the trellis node."""
        return str(self)


class Edge:
    """A directed edge in a trellis graph."""

    from_node: Node
    to_node: Node
    weight: float

    def __init__(
        self,
        from_node: Node,
        to_node: Node,
        graph_depth: int,
        *,
        generic_frequencies: bool,
    ) -> None:
        """Initialize the directed trellis edge."""
        self.from_node = from_node
        self.to_node = to_node
        # Assuming start and end vertex are not included in graph_depth.
        depth = graph_depth - 2
        word_length = (
            "*"
            if generic_frequencies or depth > NGRAM_TABLE_MAX_WORD_SIZE
            else str(depth)
        )
        if self.from_node.letter == ",":
            self.weight = int(
                onegram_tsv[self.to_node.letter][
                    unigram_notation(word_length, first_letter=True)
                ],
            )
        elif self.to_node.letter == ".":
            self.weight = int(
                onegram_tsv[self.from_node.letter][
                    unigram_notation(word_length, first_letter=False)
                ],
            )
        else:
            if word_length != "*":
                maybe_letter_positions = (
                    str(self.from_node.index),
                    str(self.to_node.index),
                )
            else:
                maybe_letter_positions = None

            self.weight = int(
                bigram_tsv[self.from_node.letter + self.to_node.letter][
                    bigram_notation(word_length, maybe_letter_positions)
                ],
            )
        if self.weight == 0:
            raise KeyError

    def __str__(self) -> str:
        """Return a string representation of the edge."""
        return f"{self.from_node} -{self.weight}-> {self.to_node}"

    def __repr__(self) -> str:
        """Return a string representation of the edge."""
        return str(self)


class Graph:
    """A trellis graph for finding number sequence do word conversions."""

    nodes: list[Node]
    edges: list[Edge]
    keypad: dict[str, list[str]]
    number_sequence: str

    def __init__(
        self,
        number_sequence: str,
        keypad: dict[str, list[str]],
        *,
        generic_frequencies: bool,
    ) -> None:
        """Initialize the trellis graph."""
        # We're representing START and STOP with "," and "."
        self.keypad = {**keypad, ",": [","], ".": ["."]}
        self.nodes = []
        self.edges = []
        if not number_sequence.isnumeric():
            err = "The provided number sequence is not numeric"
            raise ValueError(err)

        # Adding start and stop symbols.
        self.number_sequence = f",{number_sequence}."

        node_id = 0

        for index, number in enumerate(self.number_sequence):
            letters = self.keypad[number]
            self.nodes.extend(
                Node(letter, index, node_id + i) for i, letter in enumerate(letters)
            )
            node_id += len(letters)

        self.edges: list[Edge] = []

        for node in self.nodes:
            # We don't add any edges from the end node.
            if node.letter == ".":
                continue
            child_nodes = [
                child_node
                for child_node in self.nodes
                if child_node.index == node.index + 1
            ]

            for child_node in child_nodes:
                try:
                    edge = Edge(
                        node,
                        child_node,
                        len(self.number_sequence),
                        generic_frequencies=generic_frequencies,
                    )
                    self.edges.append(edge)
                except KeyError:
                    # The edge doesn't exist in the database. Just don't create it.
                    pass

    def normalized_probability(self, edge: Edge) -> float:
        """Return the normalized probability of the edge compared to its siblings."""
        # For the edges incident with the end node, we consider its siblings to be the
        # other edges incident with the end node, meaning that the to_node is the same.
        # For other edges, the from_node should be the same.
        if edge.to_node.letter == ".":
            siblings = [
                sibling
                for sibling in self.edges
                if sibling.to_node.id == edge.to_node.id
            ]
        else:
            siblings = [
                sibling
                for sibling in self.edges
                if sibling.from_node.id == edge.from_node.id
            ]
        sibling_weight = sum(sibling.weight for sibling in siblings)
        normalizing_constant = 1 / sibling_weight
        return edge.weight * normalizing_constant

    def dijkstras(self) -> str:
        """Return the string representing the shortest path in the trellis graph."""
        graph = [
            [np.inf for _ in range(len(self.nodes))] for _ in range(len(self.nodes))
        ]
        for edge in self.edges:
            norm_prob = self.normalized_probability(edge)
            # Ignore edges with probability 0
            if norm_prob == 0:
                continue

            try:
                neg_log_prob = -math.log(norm_prob)
            except ValueError:
                print(edge)
                print(norm_prob)
                raise

            graph[edge.from_node.id][edge.to_node.id] = neg_log_prob

        graph = scipy.sparse.csgraph.csgraph_from_dense(graph, null_value=np.inf)

        graph = scipy.sparse.csr_array(graph)
        _, predecessors, _ = scipy.sparse.csgraph.dijkstra(
            csgraph=graph,
            indices=0,
            min_only=True,
            return_predecessors=True,
        )

        msg = ""
        current_index = predecessors[-1]
        while current_index != 0:
            msg += self.nodes[current_index].letter
            current_index = predecessors[current_index]

        return msg[::-1]


def word_to_number_sequence(keypad: dict[str, list[str]], word: str) -> str:
    """Return the number sequence representing the provided word in the keypad."""
    num_seq = ""
    for letter in word.upper():
        for number, letters in keypad.items():
            if letter in letters:
                num_seq += number
                break

    return num_seq


KEYPAD = {
    "2": ["A", "B", "C"],
    "3": ["D", "E", "F"],
    "4": ["G", "H", "I"],
    "5": ["J", "K", "L"],
    "6": ["M", "N", "O"],
    "7": ["P", "Q", "R", "S"],
    "8": ["T", "U", "V"],
    "9": ["W", "X", "Y", "Z"],
}

KEYPAD2 = {
    "0": ["E", "T", "A"],
    "1": ["O", "I", "N"],
    "2": ["S", "R", "H"],
    "3": ["L", "D", "C"],
    "4": ["U", "M", "F"],
    "5": ["P", "G", "W"],
    "6": ["Y", "B", "V"],
    "7": ["K", "X", "J"],
    "8": ["Q", "Z"],
}

MOST_COMMON_WORDS = [
    "THE",
    "OF",
    "AND",
    "TO",
    "IN",
    "A",
    "IS",
    "THAT",
    "FOR",
    "IT",
    "AS",
    "WAS",
    "WITH",
    "BE",
    "BY",
    "ON",
    "NOT",
    "HE",
    "I",
    "THIS",
    "ARE",
    "OR",
    "HIS",
    "FROM",
    "AT",
    "WHICH",
    "BUT",
    "HAVE",
    "AN",
    "HAD",
    "THEY",
    "YOU",
    "WERE",
    "THEIR",
    "ONE",
    "ALL",
    "WE",
    "CAN",
    "HER",
    "HAS",
    "THERE",
    "BEEN",
    "IF",
    "MORE",
    "WHEN",
    "WILL",
    "WOULD",
    "WHO",
    "SO",
    "NO",
]

RANDOM_WORDS = [
    "CYCLE",
    "END",
    "BLIND",
    "HOUR",
    "FEARFUL",
    "PARK",
    "MATTER",
    "USEFUL",
    "SNATCH",
    "UTTERMOST",
    "SHEET",
    "ABAFT",
    "PUNCTURE",
    "ENORMOUS",
    "LAUNCH",
    "DUCKS",
    "CLEVER",
    "PUNISH",
    "ZANY",
    "YELLOW",
    "BEHAVE",
    "SIZE",
    "LANGUID",
    "PERIODIC",
    "NUMBER",
    "PLOT",
    "NOISE",
    "EFFICIENT",
    "CORN",
    "SYSTEM",
    "TUB",
    "BORING",
    "CONTROL",
    "TRIP",
    "SCARY",
    "MIGHTY",
    "FORTUNATE",
    "JAM",
    "TRAVEL",
    "FRIEND",
    "FALLACIOUS",
    "UNFASTEN",
    "TOUCH",
    "HORN",
    "CREEPY",
    "FLAG",
    "ADAPTABLE",
    "CRAYON",
    "PRETTY",
    "PEEP",
]


def test_common_and_random_words() -> None:
    """Test the algorithm on the 50 most common English words and 50 random words."""
    v1_common_correct_words = []
    v2_common_correct_words = []
    v3_common_correct_words = []
    v4_common_correct_words = []
    v1_common_incorrect_words = []
    v2_common_incorrect_words = []
    v3_common_incorrect_words = []
    v4_common_incorrect_words = []
    for word in MOST_COMMON_WORDS:
        string = word_to_number_sequence(KEYPAD, word)
        graph = Graph(string, KEYPAD, generic_frequencies=True)
        msg = graph.dijkstras()
        if msg == word:
            v1_common_correct_words.append(msg)
        else:
            v1_common_incorrect_words.append((word, msg))

        graph = Graph(string, KEYPAD, generic_frequencies=False)
        msg = graph.dijkstras()
        if msg == word:
            v2_common_correct_words.append(msg)
        else:
            v2_common_incorrect_words.append((word, msg))

        string = word_to_number_sequence(KEYPAD2, word)
        graph = Graph(string, KEYPAD2, generic_frequencies=True)
        msg = graph.dijkstras()
        if msg == word:
            v3_common_correct_words.append(msg)
        else:
            v3_common_incorrect_words.append((word, msg))

        graph = Graph(string, KEYPAD2, generic_frequencies=False)
        msg = graph.dijkstras()
        if msg == word:
            v4_common_correct_words.append(msg)
        else:
            v4_common_incorrect_words.append((word, msg))

    print("v1 correct common words:")
    pprint(v1_common_correct_words)
    print("v1 incorrect common words:")
    pprint(v1_common_incorrect_words)
    print("v2 correct common words:")
    pprint(v2_common_correct_words)
    print("v2 incorrect common words:")
    pprint(v2_common_incorrect_words)
    print("v3 correct common words:")
    pprint(v3_common_correct_words)
    print("v3 incorrect common words:")
    pprint(v3_common_incorrect_words)
    print("v4 correct common words:")
    pprint(v4_common_correct_words)
    print("v4 incorrect common words:")
    pprint(v4_common_incorrect_words)

    v1_random_correct_words = []
    v2_random_correct_words = []
    v3_random_correct_words = []
    v4_random_correct_words = []
    v1_random_incorrect_words = []
    v2_random_incorrect_words = []
    v3_random_incorrect_words = []
    v4_random_incorrect_words = []

    for word in RANDOM_WORDS:
        string = word_to_number_sequence(KEYPAD, word)
        graph = Graph(string, KEYPAD, generic_frequencies=True)
        msg = graph.dijkstras()
        if msg == word:
            v1_random_correct_words.append(msg)
        else:
            v1_random_incorrect_words.append((word, msg))

        graph = Graph(string, KEYPAD, generic_frequencies=False)
        msg = graph.dijkstras()
        if msg == word:
            v2_random_correct_words.append(msg)
        else:
            v2_random_incorrect_words.append((word, msg))

        string = word_to_number_sequence(KEYPAD2, word)
        graph = Graph(string, KEYPAD2, generic_frequencies=True)
        msg = graph.dijkstras()
        if msg == word:
            v3_random_correct_words.append(msg)
        else:
            v3_random_incorrect_words.append((word, msg))

        graph = Graph(string, KEYPAD2, generic_frequencies=False)
        msg = graph.dijkstras()
        if msg == word:
            v4_random_correct_words.append(msg)
        else:
            v4_random_incorrect_words.append((word, msg))

    print("v1 correct random words:")
    pprint(v1_random_correct_words)
    print("v1 incorrect random words:")
    pprint(v1_random_incorrect_words)
    print("v2 correct random words:")
    pprint(v2_random_correct_words)
    print("v2 incorrect random words:")
    pprint(v2_random_incorrect_words)
    print("v3 correct random words:")
    pprint(v3_random_correct_words)
    print("v3 incorrect random words:")
    pprint(v3_random_incorrect_words)
    print("v4 correct random words:")
    pprint(v4_random_correct_words)
    print("v4 incorrect random words:")
    pprint(v4_random_incorrect_words)

    print()
    print(
        f"results for top 50 common words: v1({len(v1_common_correct_words)}), v2({len(v2_common_correct_words)}), v3({len(v3_common_correct_words)}), v4({len(v4_common_correct_words)})",
    )
    print(
        f"results for 50 random words: v1({len(v1_random_correct_words)}), v2({len(v2_random_correct_words)}), v3({len(v3_random_correct_words)}), v4({len(v4_random_correct_words)})",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        choices=["number", "word", "test"],
        required=True,
    )
    parser.add_argument("string")
    args = parser.parse_args()

    if args.type == "test":
        test_common_and_random_words()
    else:
        number_sequcence = (
            args.string
            if args.type == "number"
            else word_to_number_sequence(KEYPAD, args.string)
        )
        graph = Graph(number_sequcence, KEYPAD, generic_frequencies=True)
        msg = graph.dijkstras()
        print(f"V1: {msg}")
        graph = Graph(number_sequcence, KEYPAD, generic_frequencies=False)
        msg = graph.dijkstras()
        print(f"V2: {msg}")

        number_sequcence = (
            args.string
            if args.type == "number"
            else word_to_number_sequence(KEYPAD2, args.string)
        )

        graph = Graph(number_sequcence, KEYPAD2, generic_frequencies=True)
        msg = graph.dijkstras()
        print(f"V3: {msg}")
        graph = Graph(number_sequcence, KEYPAD2, generic_frequencies=False)
        msg = graph.dijkstras()
        print(f"V4: {msg}")
