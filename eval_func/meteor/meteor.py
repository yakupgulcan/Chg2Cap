#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.lock = threading.Lock()
        self.meteor_p = None
        resource = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'paraphrase-en.gz')
        if not os.path.isfile(resource):
            raise FileNotFoundError('METEOR resource missing: ' + resource)
        self.env = os.environ.copy()
        self.env['LC_ALL'] = 'en_US.UTF_8'
        # Reserve stdout for METEOR's protocol; JVM diagnostics belong on stderr.
        self.meteor_cmd = ['java', '-Xlog:disable', '-Xlog:all=warning:stderr',
                           '-Duser.language=en', '-Duser.country=US', '-Xmx2G', '-jar', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE,
                                         env=self.env, universal_newlines=True, bufsize=1)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):

        scores = []

        eval_line = 'EVAL'
        with self.lock:
            for i in range(len(res)):
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            # Flush explicitly: line buffering is not guaranteed for Windows pipes.
            self.meteor_p.stdin.write(eval_line + '\n')
            self.meteor_p.stdin.flush()

            # Collect segment scores
            for i in range(len(res)):
                score = float(self._read_line())
                scores.append(score)

            # Final score
            final_score = float(self._read_line())

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line + '\n')
        self.meteor_p.stdin.flush()
        return self._read_line()

    def _read_line(self):
        line = self.meteor_p.stdout.readline()
        if not line:
            self.meteor_p.wait()
            error = self.meteor_p.stderr.read().strip()
            raise RuntimeError('METEOR Java process stopped: ' + error)
        return line.strip()

    def close(self):
        with self.lock:
            if self.meteor_p is not None:
                if self.meteor_p.poll() is None:
                    self.meteor_p.kill()
                self.meteor_p.wait()
                for stream in (self.meteor_p.stdin, self.meteor_p.stdout, self.meteor_p.stderr):
                    stream.close()
                self.meteor_p = None

    def __del__(self):
        self.close()
