import re
import numpy as np

class BaseTokenizer:
    def __init__(self, strings):
        """
            Default constructor for BaseTokenizer
        """
        
        self.strings = strings
        self.vocab_size = len(self.strings)

        # Ctrat a mapping from token to string AND string to token
        self.token_to_string = {token: string for token, string in enumerate(self.strings)}
        self.string_to_token = {string: token for token, string in enumerate(self.strings)}
        
    def itos(self, token):
        """
            Index to String! Pass the Index
            or token you want and get the string
            
            Args:
                token (int): token index

            Returns:
                token_string (str): string corresponding to token
        """
        assert 0 <= token < self.vocab_size
        return self.token_to_string[token]

    def stoi(self, string):
        """
            String to index! Pass the string
            and get the corresponding token.
            
            Args:
                string (str): string corresponding to token

            Returns:
                token (int): token index
        """
        if string in self.strings:
            return self.string_to_token[string]


class SpecialTokenizer(BaseTokenizer):
    def __init__(self):
        strings = ["<pad>", "<sos>", "<eos>", "<unk>"]
        BaseTokenizer.__init__(self, strings)
        

class NameTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "note_on", "note_off", "note_sustain",
            "pedal_on", "pedal_off", "pedal_sustain",
            "beat", "downbeat",
        ]
        strings = ["name={}".format(s) for s in strings]
        BaseTokenizer.__init__(self, strings)
        self.vocab_size = 100


class TimeTokenizer:
    def __init__(self):
        self.vocab_size = 6001
        self.frames_per_second = 100

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        time = token / self.frames_per_second
        string = "time={}".format(time)
        return string

    def stoi(self, string):
        if "time=" in string:
            time = float(re.search('time=(.*)', string).group(1))
            token = round(time * self.frames_per_second)
            return token
        
class OnsetTimeTokenizer:
    def __init__(self):
        self.vocab_size = 6001
        self.frames_per_second = 100
    
    def itos(self, token):
        assert 0 <= token < self.vocab_size

        time = token / self.frames_per_second
        string = "onset={}".format(time)
        return string

    def stoi(self, string):
        if "onset=" in string:
            time = float(re.search('onset=(.*)', string).group(1))
            token = round(time * self.frames_per_second)
            return token
        
class OffsetTimeTokenizer:
    def __init__(self):
        self.vocab_size = 6001
        self.frames_per_second = 100
    
    def itos(self, token):
        assert 0 <= token < self.vocab_size

        time = token / self.frames_per_second
        string = "offset={}".format(time)
        return string

    def stoi(self, string):
        if "offset=" in string:
            time = float(re.search('offset=(.*)', string).group(1))
            token = round(time * self.frames_per_second)
            return token

class MaestroLabelTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "Piano",
        ]
        strings = ["label=maestro-{}".format(s) for s in strings]

        BaseTokenizer.__init__(self, strings)


class Slakh2100LabelTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Ethnic",
            "Guitar",
            "Organ",
            "Percussive",
            "Piano",
            "Pipe",
            "Reed",
            "Sound Effects",
            "Strings",
            "Strings (continued)",
            "Synth Lead",
            "Synth Pad"
        ]
        strings = ["label=slakh2100-{}".format(s) for s in strings]

        BaseTokenizer.__init__(self, strings)


class GtzanLabelTokenizer(BaseTokenizer):
    def __init__(self):

        strings = ["blues", "classical", "country", "disco", "hiphop", "jazz", 
            "metal", "pop", "reggae", "rock"]

        strings = ["label=gtzan-{}".format(s) for s in strings]

        BaseTokenizer.__init__(self, strings)


class PitchTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "pitch={}".format(token)
        return string

    def stoi(self, string):
        if "pitch=" in string:
            token = int(re.search('pitch=(.*)', string).group(1))
            return token


class VelocityTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "velocity={}".format(token)
        return string

    def stoi(self, string):
        if "velocity=" in string:
            token = int(re.search('velocity=(.*)', string).group(1))
            return token


class BeatTokenizer:
    def __init__(self):
        self.vocab_size = 16

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "beat_index={}".format(token)
        return string

    def stoi(self, string):
        if "beat_index=" in string:
            token = int(re.search('beat_index=(.*)', string).group(1))
            return token


class TaskTokenizer(BaseTokenizer):
    def __init__(self):
        strings = [
            "onset", "offset", "velocity", "flatten"
        ]
        strings = ["task={}".format(s) for s in strings]
        BaseTokenizer.__init__(self, strings)
        self.vocab_size = 100

class Tokenizer:
    def __init__(self, verbose=False):
        self.tokenizers = [
            SpecialTokenizer(),
            NameTokenizer(), 
            TimeTokenizer(),
            # OnsetTimeTokenizer(),
            # OffsetTimeTokenizer(),
            MaestroLabelTokenizer(),
            Slakh2100LabelTokenizer(),
            GtzanLabelTokenizer(),
            PitchTokenizer(),
            VelocityTokenizer(),
            BeatTokenizer(),
            TaskTokenizer(),
        ]

        self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])

        if verbose:
            print("Vocab size: {}".format(self.vocab_size))
            for tokenizer in self.tokenizers:
                print(tokenizer.vocab_size)

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        for tokenizer in self.tokenizers:
            if token >= tokenizer.vocab_size:
                token -= tokenizer.vocab_size
            else:
                break
            
        return tokenizer.itos(token)

    def stoi(self, string):
        
        start_token = 0

        for tokenizer in self.tokenizers:
            
            token = tokenizer.stoi(string)
            
            if token is not None:
                return start_token + token
            else:
                start_token += tokenizer.vocab_size

        raise NotImplementedError("{} is not supported!".format(string))

    def strings_to_tokens(self, strings):

        tokens = []

        for string in strings:
            tokens.append(self.stoi(string))

        return tokens

    def tokens_to_strings(self, tokens):

        strings = []

        for token in tokens:
            strings.append(self.itos(token))

        return strings
