"""Native numeric token boundaries, refusal to guess words, and input bounds."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_sequence_frontend import encode_words
class Frontend(unittest.TestCase):
    def test_both_frontends_preserve_boundaries_and_pad_order(self):
        inventory={'a':[1,0,4,0,2],'b':[1,0,5,0,6,0,2]}
        for language in ('ru','uk'):
            self.assertEqual(encode_words(['a'],inventory,language),inventory['a'])
            self.assertEqual(encode_words(['a','b'],inventory,language),[1,0,4,0,3,0,5,0,6,0,2])
            with self.assertRaisesRegex(ValueError,'missing'):encode_words(['unknown'],inventory,language)
            with self.assertRaisesRegex(ValueError,'1024'):encode_words(['a']*400,inventory,language)
            with self.assertRaisesRegex(ValueError,'boundaries'):encode_words(['a'],{'a':[1,0,4,0,7]},language)
if __name__=='__main__':unittest.main()
