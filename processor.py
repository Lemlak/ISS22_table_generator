import numpy as np
from mido import MidiFile, MetaMessage, MidiTrack, tick2second, merge_tracks
import string
from tqdm import tqdm
import mido
import time
from scipy import signal
import sys

from itertools import chain
def longlist2array2(top_list):
    return np.array([np.fromiter(chain.from_iterable(longlist), np.array(longlist[0][0]).dtype, -1).reshape((len(longlist), -1)) for longlist in top_list])

def get_segments(array):
    array = np.concatenate([[np.infty], array, [np.infty]])
    deltas = array[1:]-array[:-1]
    dw = np.where(deltas!=0)[0]
    return zip(dw[:-1], dw[1:])
    
def _to_abstime(messages):
    """Convert messages to absolute time."""
    now = 0
    for msg in messages:
        now += msg.time
        yield msg.copy(time=now)


def _to_reltime(messages):
    """Convert messages to relative time."""
    now = 0
    for msg in messages:
        delta = msg.time - now
        yield msg.copy(time=delta)
        now = msg.time


def fix_end_of_track(messages):
    """Remove all end_of_track messages and add one at the end.
    This is used by merge_tracks() and MidiFile.save()."""
    # Accumulated delta time from removed end of track messages.
    # This is added to the next message.
    accum = 0

    for msg in messages:
        if msg.type == 'end_of_track':
            accum += msg.time
        else:
            if accum:
                delta = accum + msg.time
                yield msg.copy(time=delta)
                accum = 0
            else:
                yield msg

    yield MetaMessage('end_of_track', time=accum)

class MyMidiFile(MidiFile):
    def __init__(self, midi_file):
        super().__init__(midi_file)

    def __iter__(self):
        # The tracks of type 2 files are not in sync, so they can
        # not be played back like this.
        if self.type == 2:
            raise TypeError("can't merge tracks in type 2 (asynchronous) file")

        tempo = 500000
        num = 4
        den = 4
        for msg in merge_tracks(self.tracks):
            # Convert message time from absolute time
            # in ticks to relative time in seconds.
            if msg.time > 0:
                delta = tick2second(msg.time, self.ticks_per_beat, tempo) 
            else:
                delta = 0

            yield msg.copy(time=delta)

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.type == 'time_signature':
                num = msg.numerator
                den = msg.denominator
                
                
class ISSMidiFile():
    def __init__(self, midi_file, verbosity=0, mtype=None):
        self.tempo = 500000
        self.numerator = 4
        self.denominator = 4
        self.loaded_midi = MyMidiFile(midi_file)
        if mtype is not None:   
            run_type = mtype
        else:
            run_type = self.loaded_midi.type
        if run_type == 1:
            self.from_mido()
        else:
            self.tpb = self.loaded_midi.ticks_per_beat
            self.verbosity = verbosity
            self.note_mat = self.mid2arry(self.loaded_midi)
            self.opt_arry2table()

    def from_mido(self):
        input_time = 0.0
        time_steps = []
        note = [0]*90
        i = 0
        for msg in self.loaded_midi:
            if msg.time > 0:
                note[-1] = input_time
                input_time += msg.time
                note[-2] = input_time
                time_steps.append(note.copy())
                #print(note)
            #print(msg.note, msg.velocity)   
            if msg.type == 'note_off':
                note[msg.note-21] = 0
            if msg.type == 'note_on':
                note[msg.note-21] = msg.velocity
        #print(time_steps)
        
        time_steps = np.array(time_steps)
        for midi in np.arange(88):
            if np.sum(time_steps[:,midi]) > 0:
                deltas = time_steps[1:,midi]-time_steps[:-1,midi]
                dw = np.where(deltas!=0)[0]+1
                segments = dw[1:] - dw[:-1]
                for i, s in enumerate(segments):
                    if sum(time_steps[:,midi][dw[i]:dw[i]+s]) >0:
                        print(time_steps[dw[i],89]*1000, time_steps[dw[i]+s,88]*1000, midi+21, int(time_steps[dw[i],midi]))
         
    def opt_arry2table(self):
        for track in tqdm(self.note_mat, desc='printing'):
            for den_seg in get_segments(track[:,-1]):
                den = track[den_seg[0],-1]
                for num_seg in get_segments(track[den_seg[0]:den_seg[1],-2]):
                    num = track[num_seg[0],-2]
                    for tempo_seg in get_segments(track[num_seg[0]:num_seg[1],-3]):
                        tempo = track[num_seg[0],-3]
                        for m in np.arange(88):
                            for midi_seg in get_segments(track[tempo_seg[0]:tempo_seg[1],m]):
                                if np.sum(track[midi_seg[0]:midi_seg[1],m]) > 0:
                                    #print(tempo)
                                    start_t = 1000*mido.tick2second(midi_seg[0], self.tpb, tempo)* num/den
                                    end_t = 1000*mido.tick2second(midi_seg[1], self.tpb, tempo)* num/den
                                    print(start_t, end_t, m+21, track[midi_seg[0],m])


    # these functions are taken and modified from https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    def msg2dict(self, msg):
        result = dict()
        if 'note_on' in msg:
            on_ = True
        elif 'note_off' in msg:
            on_ = False
        else:
            on_ = None
        
        result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
            str.maketrans({a: None for a in string.punctuation})))

        if on_ is not None:
            for k in ['note', 'velocity']:
                result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                    str.maketrans({a: None for a in string.punctuation})))
        return [result, on_]

    def switch_note(self, last_state, note, velocity, on_=True):
        # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
        result = [0] * 91 if last_state is None else last_state.copy()
        if 21 <= note <= 108:
            result[note-21] = velocity if on_ else 0
            result[-3], result[-2], result[-1] = self.tempo, self.numerator, self.denominator
        return result

    def get_new_state(self, new_msg, last_state):
        if type(new_msg) == MetaMessage and new_msg.type == 'set_tempo':
            self.tempo = new_msg.tempo
            if self.verbosity > 1:
                print('Setting tempo', new_msg)
        if type(new_msg) == MetaMessage and new_msg.type == 'time_signature':
            if self.verbosity > 1:
                print('Setting time signature from', self.numerator, self.denominator)
            self.numerator = new_msg.numerator
            self.denominator = new_msg.denominator
            if self.verbosity > 1:
                print('Setting time signature to', self.numerator, self.denominator)
                print(new_msg)
        elif type(new_msg) == MetaMessage:
            if self.verbosity > 4:
                print(new_msg)
        new_msg, on_ = self.msg2dict(str(new_msg))
        new_state = self.switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
        return [new_state, new_msg['time']]

    def track2seq(self, track):
        # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
        # 89, 90 and 91 are reserved for tempo, numerator and denominator respectively
        result = []
        first_state = [0]*91
        first_state[-3], first_state[-2], first_state[-1] = 500000, 4, 4
        last_state, last_time = self.get_new_state(str(track[0]), first_state)
        for i in range(1, len(track)):
            new_state, new_time = self.get_new_state(track[i], last_state)
            if new_time > 0:
                #last_state[-3], last_state[-2], last_state[-1] = self.tempo, self.numerator, self.denominator
                #print(self.tempo)
                result += [last_state]*new_time
            last_state, last_time = new_state, new_time
        return result

    def mid2arry(self, mid, min_msg_pct=0.0):
        tracks_len = [len(tr) for tr in mid.tracks]
        min_n_msg = max(tracks_len) * min_msg_pct
        if self.verbosity > 3:
            print(f'Number of tracks: {len(mid.tracks)}')
        
        # convert each track to nested list
        all_arys = []
        for i in tqdm(range(len(mid.tracks)), desc='loading tracks'):
            if self.verbosity > 3: 
                print(f'Track: {i}')
            if len(mid.tracks[i]) > min_n_msg:
                ary_i = self.track2seq(mid.tracks[i])
                all_arys.append(ary_i)
        # make all nested list the same length
        max_len = max([len(ary) for ary in all_arys])
        for i in tqdm(range(len(all_arys)), desc='trimming'):
            if len(all_arys[i]) < max_len:
                blank_state  = [0]*91
                blank_state[-3], blank_state[-2], blank_state[-1] = self.tempo, self.numerator, self.denominator
                all_arys[i] += [blank_state] * (max_len - len(all_arys[i]))
        all_arys = longlist2array2(all_arys)
        return all_arys

