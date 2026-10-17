"""Numeric contexts must retain each target word's entire token span."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_latent_export import numeric_context,phrase_tokens

class Context(unittest.TestCase):
    def test_all_contexts_locate_target_without_neighbor_tokens(self):
        words='zero one two four twenty hundred thousand million plus minus times divided by equals negative positive point open close parenthesis'.split()
        inventory={w:[0,i+1,0,i+100,0] for i,w in enumerate(words)}
        for word in words:
            phrase,index=numeric_context(word)
            self.assertEqual(phrase[index],word)
            encoded,start,end=phrase_tokens(phrase,index,inventory,500)
            self.assertEqual(encoded[start:end],inventory[word][1:])
            self.assertEqual(encoded[0],0)
            if start>1:self.assertEqual(encoded[start-2:start],[500,0])
            if end<len(encoded):self.assertEqual(encoded[end:end+2],[500,0])
    def test_grammatical_units_signs_and_division(self):
        self.assertEqual(numeric_context('million'),(['one','million','plus','one'],1))
        self.assertEqual(numeric_context('negative'),(['negative','one','plus','one'],0))
        self.assertEqual(numeric_context('by'),(['one','divided','by','one'],2))

if __name__=='__main__':unittest.main()
