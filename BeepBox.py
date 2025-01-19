import math, os
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment

def splitString(s: str):
    return [s[i:i + 5] for i in range(0, len(s), 5)]

def midi_to_frequency(midi_note: int) -> float:
    return (440 / 32) * (2 ** ((midi_note - 9) / 12))

def lerp(a, b, t):
    return a + (b-a)*t

def trig_interpolation(a, b, t):
    t = math.sin(t*math.pi/2)**2
    return lerp(a, b, t)

def add_arrays_at_indices(base_array, array, index):
    result_array = base_array.copy()
    if index + len(array) > len(result_array):
        raise ValueError(f"The array starting at index {index} cannot fit in the base array.")
    result_array[index:index + len(array)] += array
    return result_array

class BeepBox:
    def __init__(self, path, debug:bool = True):
        self.path = path
        self.debug = debug
        if os.path.exists(path):
            with open(path, "r") as f:
                self.TSCode = self.loadCode(f.read())
        else:
            print(f"Path: {path} doesn't exist")
            del self


    def loadCode(self, code):
        tempo = float(code.split("!")[0])
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        data = code.split("!")[1:-1]
        noteData = []
        for i in data:
            k = splitString(i)
            notesChunk = []
            for j in k:
                if len(j)==5:
                    if j[2] != "l":
                        note = int(j[0]+j[1])+14
                        volume = float(chars.index(j[4])/(len(chars)-1))
                        length = chars.index(j[3])
                        notesChunk.append([note, volume, length, chars.index(j[2])%4])
            noteData.append(notesChunk)
        return {"tempo": tempo, "data": noteData}

    def _getMaxTime(self, bd):
        t = 0
        ct = 0
        for i in self.TSCode["data"]:
            for j in i:
                h = ct + bd*(j[2]+1)
                if h > t:
                    t = h
            ct += bd
        return t

    def getMaxVolumeAudio(self):
        volumes = []
        for num in range(len(self.TSCode["data"])):
            vol = 0
            for j in range(num+1):
                for k in self.TSCode["data"][j]:
                    if j+k[2]>=num:
                        vol += k[1]
            volumes.append(vol)
        mv = max(volumes)
        return mv

    def getTotalNotes(self):
        p = 0
        for i in self.TSCode["data"]:
            p += len(i)
        return p

    def getAmplitudeArray(self, volume, beats, interpolation_rules, beat_duration, sample_rate, maxVolume):
        a = self.maxAmplitudeWav * volume / maxVolume
        l = int(sample_rate * interpolation_rules[0] * beat_duration)
        amplitude = np.array([trig_interpolation(0, a, t / l) for t in range(l)])
        l = int(sample_rate * (beats + 1 - interpolation_rules[1] - interpolation_rules[0]) * beat_duration)
        amplitude = np.concatenate((amplitude, a*np.ones(l)))
        l = int(sample_rate * interpolation_rules[1] * beat_duration)
        return np.concatenate((amplitude, np.array([trig_interpolation(a, 0, t / l) for t in range(l)])))


    def customWave(self, noteData, interpolation_rules, beat_duration, time_start, sample_rate, maxVolume):
        amplitude = self.getAmplitudeArray(noteData[1], noteData[2], interpolation_rules, beat_duration, sample_rate, maxVolume)
        time = np.linspace(time_start, time_start + (noteData[2]+1) * beat_duration, len(amplitude), endpoint=False)
        freq = midi_to_frequency(noteData[0])
        if noteData[3] == 0:
            """  Sine Wave  (TuneShare Instrument: Piano)  """
            wave = amplitude * np.sin(2 * np.pi * freq * time)
        elif noteData[3] == 1:
            """  Square Wave  (TuneShare Instrument: Electric Piano)  """
            wave = amplitude * np.array([0 if t == 0 else t / abs(t) for t in np.sin(2 * np.pi * freq * time)])
        elif noteData[3] == 2:
            """  Triangle Wave  (TuneShare Instrument: Organ)  """
            wave = amplitude * np.array([-1+4*((freq*t)%1) if (freq*t)%1<=0.5 else 3-4*((freq*t)%1) for t in time])
        elif noteData[3] == 3:
            """  Sawtooth Wave  (TuneShare Instrument: Guitar)  """
            wave = amplitude * np.array([-1+2*((freq*t)%1) for t in time])
        return wave

    def GenerateMP3(self, sample_rate=44100, export_file="Export.mp3"):
        self.sample_rate = sample_rate
        self.maxAmplitudeWav = 32767
        maxVolume = self.getMaxVolumeAudio()

        beat_duration = 15 / (self.TSCode["tempo"])
        max_time = self._getMaxTime(beat_duration)

        interpolation_start = 0.25
        interpolation_end = 0.5
        totalNotes = self.getTotalNotes()
        currentNote = 0

        self.WavData = np.zeros(int(sample_rate * max_time))
        time_start = 0
        for i in self.TSCode["data"]:
            for j in i:
                currentNote += 1
                if self.debug:
                    print(f"Working on Note: {currentNote}/{totalNotes}. (Type: {['Sine', 'Square', 'Triangle', 'Sawtooth'][j[3]]})")
                start_idx = int(time_start*sample_rate)
                wave = self.customWave(j, [interpolation_start, interpolation_end], beat_duration, time_start, sample_rate, maxVolume)
                self.WavData = add_arrays_at_indices(self.WavData, wave, start_idx)
            time_start += beat_duration
        write("Test.wav", sample_rate, self.WavData.astype(np.int16))
        self.convert_wav_to_mp3("Test.wav", export_file)
        self.delete_wav_file("Test.wav")

    def convert_wav_to_mp3(self, wav_file_path, mp3_file_path):
        try:
            audio = AudioSegment.from_wav(wav_file_path)
            audio.export(mp3_file_path, format="mp3")
            print(f"Conversion successful: {wav_file_path} -> {mp3_file_path}")
        except Exception as e:
            print(f"Error during conversion: {e}")

    def delete_wav_file(self, file_path):
        try:
            os.remove(file_path)
            print(f"File {file_path} has been deleted successfully.")
        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except PermissionError:
            print(f"You do not have permission to delete the file {file_path}.")
        except Exception as e:
            print(f"Error occurred while deleting the file: {e}")



if __name__ == "__main__":
    beepbox = BeepBox("TS_FamilyGuy.txt", debug=True)
    beepbox.GenerateMP3()
