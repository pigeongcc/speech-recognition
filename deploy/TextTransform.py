default_char_map_str = """
' 0
<BLANK> 1
a 2
b 3
c 4
d 5
e 6
f 7
g 8
h 9
i 10
j 11
k 12
l 13
m 14
n 15
o 16
p 17
q 18
r 19
s 20
t 21
u 22
v 23
w 24
x 25
y 26
z 27
"""


class TextTransform:
    """ Maps characters to their indices, and vice versa """
    def __init__(self, char_map_str: str = default_char_map_str):
        self.char_map = {}
        self.index_map = {}
        self.blank_label = None
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
            if not self.blank_label and ch == '<BLANK>':
                self.blank_label = int(index)
        self.index_map[self.blank_label] = ' '

    def text_to_int(self, text: list[str]):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ind = self.char_map['<BLANK>']
            else:
                ind = self.char_map[c]
            int_sequence.append(ind)
        return int_sequence

    def int_to_text(self, labels: list[int]):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<BLANK>', ' ').strip()