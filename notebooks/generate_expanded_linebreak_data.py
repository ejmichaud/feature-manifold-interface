"""
Generate an expanded linebreak dataset with ~5000 sequences at width=150.

Downloads ~40 Project Gutenberg books, cleans them, splits into paragraphs,
shuffles paragraphs across books (to break topic correlations), groups into
sequences, wraps at the target width, and trims a random number of lines from
the front of each sequence (to randomize paragraph boundary alignment with
line numbers). Saves using the same \n\n=====\n\n separator as existing data.

Usage:
    python notebooks/generate_expanded_linebreak_data.py
    python notebooks/generate_expanded_linebreak_data.py --width 150 --num-sequences 5000
    python notebooks/generate_expanded_linebreak_data.py --output-file notebooks/LineBreakManifold/linebreak_data/linebreak_width_150_expanded.txt
"""

import argparse
import sys
import random
from pathlib import Path

# Add LineBreakManifold directory to path so we can import utilities
sys.path.insert(0, str(Path(__file__).parent / "LineBreakManifold"))
from linebreak_utils import download_gutenberg_book, clean_gutenberg_text, wrap_text_to_width


BOOK_IDS = [
    # Original 10
    1342,  # Pride and Prejudice - Jane Austen
    11,    # Alice's Adventures in Wonderland - Lewis Carroll
    1661,  # The Adventures of Sherlock Holmes - Arthur Conan Doyle
    84,    # Frankenstein - Mary Shelley
    2701,  # Moby Dick - Herman Melville
    1952,  # The Yellow Wallpaper - Charlotte Perkins Gilman
    158,   # Emma - Jane Austen
    345,   # Dracula - Bram Stoker
    98,    # A Tale of Two Cities - Charles Dickens
    174,   # The Picture of Dorian Gray - Oscar Wilde
    # Additional ~30 books
    1260,  # Jane Eyre - Charlotte Bronte
    768,   # Wuthering Heights - Emily Bronte
    2554,  # Crime and Punishment - Fyodor Dostoevsky
    1080,  # A Modest Proposal - Jonathan Swift
    46,    # A Christmas Carol - Charles Dickens
    219,   # Heart of Darkness - Joseph Conrad
    74,    # The Adventures of Tom Sawyer - Mark Twain
    76,    # Adventures of Huckleberry Finn - Mark Twain
    120,   # Treasure Island - Robert Louis Stevenson
    16,    # Peter Pan - J.M. Barrie
    1232,  # The Prince - Niccolo Machiavelli
    5200,  # Metamorphosis - Franz Kafka
    244,   # A Study in Scarlet - Arthur Conan Doyle
    1400,  # Great Expectations - Charles Dickens
    2591,  # Grimm's Fairy Tales - Brothers Grimm
    1184,  # The Count of Monte Cristo - Alexandre Dumas
    3207,  # Leviathan - Thomas Hobbes
    996,   # Don Quixote - Miguel de Cervantes
    6130,  # The Iliad - Homer
    1727,  # The Odyssey - Homer
    55,    # The Wonderful Wizard of Oz - L. Frank Baum
    1998,  # Thus Spake Zarathustra - Friedrich Nietzsche
    4300,  # Ulysses - James Joyce
    100,   # The Complete Works of William Shakespeare
    27827, # The Kama Sutra - Vatsyayana
    514,   # Little Women - Louisa May Alcott
    2542,  # A Doll's House - Henrik Ibsen
    43,    # The Strange Case of Dr. Jekyll and Mr. Hyde - R.L. Stevenson
    1497,  # Republic - Plato
    2600,  # War and Peace - Leo Tolstoy
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expanded linebreak dataset")
    parser.add_argument("--output-file",
                        default="notebooks/LineBreakManifold/linebreak_data/linebreak_width_150_expanded.txt",
                        help="Output file path")
    parser.add_argument("--num-sequences", type=int, default=5000,
                        help="Target number of sequences")
    parser.add_argument("--width", type=int, default=150,
                        help="Line width for wrapping")
    parser.add_argument("--min-seq-length", type=int, default=500,
                        help="Minimum sequence length in characters")
    args = parser.parse_args()

    # Download and clean all books
    print(f"Downloading and cleaning {len(BOOK_IDS)} books...")
    all_paragraphs = []
    for book_id in BOOK_IDS:
        try:
            print(f"  Book {book_id}...", end=" ", flush=True)
            text = download_gutenberg_book(book_id)
            cleaned = clean_gutenberg_text(text)
            paras = [p for p in cleaned.split("\n\n") if len(p) >= args.min_seq_length]
            all_paragraphs.extend(paras)
            print(f"OK ({len(paras)} paragraphs, {len(cleaned):,} chars)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nTotal paragraphs: {len(all_paragraphs)}")

    # Shuffle paragraphs across all books to break book-level topic correlations.
    # Without this, consecutive paragraphs from the same book would be grouped
    # together, creating sequences with coherent topics that could confound the
    # position signal we're trying to isolate.
    random.seed(42)
    random.shuffle(all_paragraphs)

    # Group shuffled paragraphs into sequences
    paras_per_seq = max(1, len(all_paragraphs) // args.num_sequences)
    sequences = []
    for i in range(0, len(all_paragraphs), paras_per_seq):
        chunk = all_paragraphs[i : i + paras_per_seq]
        seq_text = "\n\n".join(chunk)
        if len(seq_text) >= args.min_seq_length:
            sequences.append(seq_text)

    if len(sequences) > args.num_sequences:
        sequences = random.sample(sequences, args.num_sequences)

    print(f"Created {len(sequences)} sequences")
    if sequences:
        lengths = [len(s) for s in sequences]
        print(f"  Avg length: {sum(lengths) // len(lengths)} chars")
        print(f"  Range: {min(lengths)} - {max(lengths)} chars")

    # Wrap at target width, then trim a random amount from the front of each
    # sequence. This randomizes where paragraph boundaries fall relative to
    # line numbers, preventing systematic correlation between line_number and
    # paragraph-initial content (topic sentences, etc.).
    print(f"Wrapping at width={args.width} with random front trim...")
    wrapped = []
    for seq in sequences:
        w = wrap_text_to_width(seq, args.width)
        lines = w.split("\n")
        # Trim 0 to 5 lines from the front (uniformly random)
        trim = random.randint(0, min(5, max(0, len(lines) - 3)))
        if trim > 0:
            lines = lines[trim:]
        wrapped.append("\n".join(lines))

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n=====\n\n".join(wrapped), encoding="utf-8")
    file_size_kb = output_path.stat().st_size / 1024
    print(f"Saved {len(wrapped)} sequences to {args.output_file} ({file_size_kb:.1f} KB)")
