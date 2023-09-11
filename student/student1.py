from typing import List
import itertools
import matplotlib.pyplot as plt

# RobustMPC algorithm based on paper
# by Zane Middaugh

# Adapted from code by Zach Peats

# ======================================================================================================================
# Do not touch the client message class!
# ======================================================================================================================

class ClientMessage:
	"""
	This class will be filled out and passed to student_entrypoint for your algorithm.
	"""
	total_seconds_elapsed: float	  # The number of simulated seconds elapsed in this test
	previous_throughput: float		  # The measured throughput for the previous chunk in kB/s

	# not supposed to be used - buffer_current_fill: float		    # The number of kB currently in the client buffer
	buffer_seconds_per_chunk: float     # Number of seconds that it takes the client to watch a chunk. Every
										# buffer_seconds_per_chunk, a chunk is consumed from the client buffer.
	buffer_seconds_until_empty: float   # The number of seconds of video left in the client buffer. A chunk must
										# be finished downloading before this time to avoid a rebuffer event.
	buffer_max_size: float              # The maximum size of the client buffer. If the client buffer is filled beyond
										# maximum, then download will be throttled until the buffer is no longer full

	# The quality bitrates are formatted as follows:
	#
	#   quality_levels is an integer reflecting the # of quality levels you may choose from.
	#
	#   quality_bitrates is a list of floats specifying the number of kilobytes the upcoming chunk is at each quality
	#   level. Quality level 2 always costs twice as much as quality level 1, quality level 3 is twice as big as 2, and
	#   so on.
	#       quality_bitrates[0] = kB cost for quality level 1
	#       quality_bitrates[1] = kB cost for quality level 2
	#       ...
	#
	#   upcoming_quality_bitrates is a list of quality_bitrates for future chunks. Each entry is a list of
	#   quality_bitrates that will be used for an upcoming chunk. Use this for algorithms that look forward multiple
	#   chunks in the future. Will shrink and eventually become empty as streaming approaches the end of the video.
	#       upcoming_quality_bitrates[0]: Will be used for quality_bitrates in the next student_entrypoint call
	#       upcoming_quality_bitrates[1]: Will be used for quality_bitrates in the student_entrypoint call after that
	#       ...
	#
	quality_levels: int
	quality_bitrates: List[float]
	upcoming_quality_bitrates: List[List[float]]

	# You may use these to tune your algorithm to each user case! Remember, you can and should change these in the
	# config files to simulate different clients!
	#
	#   User Quality of Experience =    (Average chunk quality) * (Quality Coefficient) +
	#                                   -(Number of changes in chunk quality) * (Variation Coefficient)
	#                                   -(Amount of time spent rebuffering) * (Rebuffering Coefficient)
	#
	#   *QoE is then divided by total number of chunks
	#
	quality_coefficient: float
	variation_coefficient: float
	rebuffering_coefficient: float
# ======================================================================================================================


# Your helper functions, variables, classes here. You may also write initialization routines to be called
# when this script is first imported and anything else you wish.
past_qual = -1
harmonic_window = []
idxs = []

def enumerateSequences(curr,future):

	# should return numlevels ^ 5 length list of 5 item lists
	combined = []

	combined.append(curr)

	for f in future:
		combined.append(f)

	enum_rates = list(itertools.product(*combined))

	return enum_rates

def adjustHarmonicMean(prev_tp):
	# this is what distinguishes robust MPC

	global harmonic_window

	if prev_tp == 0:
		return -1

	if len(harmonic_window) == 5:
		harmonic_window.pop(0)

	harmonic_window.append(prev_tp)

	# calculate harmonic mean
	sum_den = 0

	for pane in harmonic_window:
		sum_den += (1 / pane)

	hmean = len(harmonic_window) / sum_den

	max = 0
	for pane in harmonic_window:
		curr = abs((hmean - pane)/hmean)

		if curr > max:
			max = curr

	err = max

	low_bound = hmean / (1 + err)

	return(low_bound)




def calcQoe(seq,m,bool):
	global past_qual

	# calculate the average quality in the sequence
	num_chunks = 0
	sum_qual = 0
	all_lvls = []
	all_sizes = []

	for i in range(len(seq)):
		num_chunks+=1

		if i == 0:
			idx = m.quality_bitrates.index(seq[i])
			curr_size = (idx+1) * seq[i] * (2 ** (idx))
		else:
			idx = m.upcoming_quality_bitrates[i-1].index(seq[i])
			curr_size = (idx+1) * seq[i] * (2 ** (idx))

		all_sizes.append(curr_size)
		all_lvls.append(idx)
		sum_qual += curr_size

	
	qoe_qual = all_sizes[0] * m.quality_coefficient

	# calculate the number of changes in quality
	past = past_qual
	num_changes = 0

	for a in all_lvls:
		if a != past:
			num_changes += 1
			past = a
	
	qoe_change = num_changes * m.variation_coefficient
	
	# calculate the rebuffering time for the sequence
	chunk_duration = m.buffer_seconds_per_chunk
	buf_state = m.buffer_seconds_until_empty
	buf_max = m.buffer_max_size
	prev_tp = m.previous_throughput

	# get the low bound calculated from the harmonic mean of last 5 throughputs
	low_bnd = adjustHarmonicMean(prev_tp)

	wait_time = 0

	for a in all_sizes:
		buf_state -= (a/low_bnd)

		if(buf_state < 0):
			wait_time += (-1*buf_state)
			buf_state = 0
		elif(buf_state > buf_max):
			buf_state = buf_max - chunk_duration
		
		buf_state += chunk_duration

	qoe_rebuf = wait_time * m.rebuffering_coefficient

	# if bool == 1:
	# 	print(f'selected seq {seq}')
	# 	print(f'with {wait_time} rebuf')
	# 	print(f'with {num_changes} changes')
	# 	print(f'with {all_sizes} quality')

	return qoe_qual - qoe_change - qoe_rebuf

	

def findBestSeq(seqs,m):
	
	for i in range(len(seqs)):
		curr_qoe = calcQoe(seqs[i],m,0)

		if i == 0:
			max_qoe = curr_qoe
			max_ind = i
		elif curr_qoe > max_qoe:
			max_qoe = curr_qoe
			max_ind = i
	
	calcQoe(seqs[max_ind],m,1)

	return seqs[max_ind]

ct = 0
prev_tps = []
bsec = []
def student_entrypoint(client_message: ClientMessage):
	global ct
	global past_qual
	
	c = client_message
	
	"""
	Your mission, if you choose to accept it, is to build an algorithm for chunk bitrate selection that provides
	the best possible experience for users streaming from your service.

	Construct an algorithm below that selects a quality for a new chunk given the parameters in ClientMessage. Feel
	free to create any helper function, variables, or classes as you wish.

	Simulation does ~NOT~ run in real time. The code you write can be as slow and complicated as you wish without
	penalizing your results. Focus on picking good qualities!

	Also remember the config files are built for one particular client. You can (and should!) adjust the QoE metrics to
	see how it impacts the final user score. How do algorithms work with a client that really hates rebuffering? What
	about when the client doesn't care about variation? For what QoE coefficients does your algorithm work best, and
	for what coefficients does it fail?

	Args:
		client_message : ClientMessage holding the parameters for this chunk and current client state.

	:return: float Your quality choice. Must be one in the range [0 ... quality_levels - 1] inclusive.
	"""

	# check if 1st chunk, if so use lowest bitrate
	if c.previous_throughput == 0:
		past_qual = 0
		return 0

	# use a window size of 5 to enumerate all possible bitrates

	seqs = enumerateSequences(c.quality_bitrates,c.upcoming_quality_bitrates[:4])

	# get the best QoE sequence
	best_qoe_plan = findBestSeq(seqs,c)

	curr_rate = best_qoe_plan[0]

	# find index of optimal bitrate
	idx = c.quality_bitrates.index(curr_rate)

	past_qual = idx


# uncomment below code to see graph
	# global idxs
	# idxs.append(idx)
	# global prev_tps
	# prev_tps.append(c.previous_throughput)
	# global bsec
	# bsec.append(c.buffer_seconds_until_empty)
	# ct+=1
	# if ct == 238:
	# 	# fig,ax = plt.subplots()
	# 	# ax.plot(range(len(idxs)),idxs,linewidth=2.0)
	# 	# ax.plot(range(len(idxs)),prev_tps,linewidth=2.0)
	# 	# ax.plot(range(len(idxs)),bsec,linewidth=2.0)
	# 	# ax.set(xlim=(0, 239),
	# 	# ylim=(0, 3), yticks=range(1, 30))
	# 	# plt.show()
	# 	with open('res1','a') as r:
	# 		r.write(str(sum(bsec)/len(bsec))+'\n')



	return idx  # return MPC selected bitrate