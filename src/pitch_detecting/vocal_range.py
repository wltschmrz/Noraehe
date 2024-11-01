# <src/pitch_detecting/vocal_range.py>

class VocalRange:
    """
    Pitch 범위를 다루는 클래스
    """
    note_to_midi = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }

    def __init__(self, min_note, max_note):
        self.min_note = min_note
        self.max_note = max_note
        self.min_midi = self.note_str_to_midi(min_note)
        self.max_midi = self.note_str_to_midi(max_note)

    def note_str_to_midi(self, note_str):
        note = note_str[:-1]
        octave = int(note_str[-1])
        return (octave + 1) * 12 + self.note_to_midi[note]

    @staticmethod
    def midi_to_note_name(midi_value):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = int((midi_value // 12) - 1)
        note = note_names[int(midi_value % 12)]
        return f"{note}{octave}"

    def get_range(self):
        return (self.min_note, self.max_note)

    def get_midi_range(self):
        return (self.min_midi, self.max_midi)

class KeyShiftCalculator:
    """
    song, user range 순서대로 넣으면 key shift 값과 octave shift 값을 계산함.
    input:
        song range (str, str)
        user range (str, str)
    output:
        semitone shift (int)
        octaves shifted (int)    
    """
    note_to_midi = VocalRange.note_to_midi

    def note_str_to_midi(self, note_str):
        note = note_str[:-1]
        octave = int(note_str[-1])
        return (octave + 1) * 12 + self.note_to_midi[note]

    def calculate_key_shift(self, song_range, user_range):
        song_low, song_high = [self.note_str_to_midi(note) for note in song_range]
        user_low, user_high = [self.note_str_to_midi(note) for note in user_range]
        
        assert song_high >= song_low and user_high >= user_low, "Range should be listed in right order: '(low, high)'"

        how_high = song_high - user_high
        total_shift = -how_high  # The initial required shift

        # Calculate the semitone shift within -6 to 6
        s = ((total_shift + 6) % 12) - 6

        # Calculate the number of octaves shifted
        k = (total_shift - s) // 12

        return s, k