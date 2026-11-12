"""Internal duplicate filtering only; no downstream benchmark access."""
import hashlib
import json
import unicodedata


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def document_sha(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def window_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def shingles(ids):
    return {tuple(ids[i:i+5]) for i in range(len(ids)-4)}


class Deduplicator:
    def __init__(self):
        self.documents = set()
        self.windows = set()
        self.signatures = []
        self.inverted = {}
        self.rejected = {'document_exact': 0, 'window_exact': 0, 'window_near': 0}

    def accept(self, text_hash, ids):
        digest = window_sha(ids)
        if text_hash in self.documents:
            self.rejected['document_exact'] += 1
            return False
        if digest in self.windows:
            self.rejected['window_exact'] += 1
            return False
        current = shingles(ids[1:-1])
        # Candidate lookup is exact: a Jaccard >= .8 pair must share a shingle.
        candidates = set()
        for item in current:
            candidates.update(self.inverted.get(item, ()))
        for index in candidates:
            prior = self.signatures[index]
            if len(current & prior)/len(current | prior) >= .8:
                self.rejected['window_near'] += 1
                return False
        index = len(self.signatures)
        self.signatures.append(current)
        for item in current:
            self.inverted.setdefault(item, []).append(index)
        self.documents.add(text_hash); self.windows.add(digest)
        return True


def reservoir(stream, size, generator):
    result = []
    eligible = 0
    scanned = 0
    for line_index, item in enumerate(stream):
        scanned += 1
        text = item['text']
        if not 500 <= len(text) <= 100000:
            continue
        eligible += 1
        row = (line_index, text)
        if len(result) < size:
            result.append(row)
        else:
            position = generator.randrange(eligible)
            if position < size:
                result[position] = row
    generator.shuffle(result)
    return result, scanned, eligible
