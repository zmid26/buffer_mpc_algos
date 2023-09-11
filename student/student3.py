from typing import List
import matplotlib.pyplot as plt

# BBA-2 algorithm based on paper
# by Zane Middaugh

# Adapted from code by Zach Peats

# ======================================================================================================================
# Do not touch the client message class!
# ======================================================================================================================


past_qual = -1
harmonic_window = []
idxs = []
ct = 0
prev_tps = []
bsec = []

def minGreater(arr,key):
	min_val = max(arr)

	for i in arr:
		if i > key and i < min_val:
			min_val = i
	
	return min_val


def maxLesser(arr,key):
	max_val = min(arr)

	for i in arr:
		if i < key and i > max_val:
			max_val = i
	
	return max_val

def chunkMap(curr_buf,max_buf,rate_max,rate_min,map_range):

	# use interval [a,b] to perform actual mapping

	a,b = map_range
	
	if curr_buf <= a:
		fb = rate_min
	elif curr_buf >= b:
		fb = rate_max
	else:
		# use a straight linear map
		fb = (curr_buf / max_buf) * (rate_max - rate_min)

	return fb


def BBA2(rate_prev,curr_buf,resv_size,cush_size,map_range,max_buf,m):
	
	vid_rates = range(3)
	max_rate = max(vid_rates)
	min_rate = min(vid_rates)
	rmap = chunkMap(curr_buf,max_buf,max_rate,min_rate,map_range)

	if rate_prev == max_rate:
		chunk_plus = max_rate
	else:
		chunk_plus = minGreater(vid_rates,rate_prev)
	
	if rate_prev == min_rate:
		chunk_minus = min_rate
	else:
		chunk_minus = maxLesser(vid_rates,rate_prev)

	# could be improved
	next_tp = m.previous_throughput

	max_chunk_rate = max(m.quality_bitrates)
	
	if len(m.upcoming_quality_bitrates) > 0:
		next_max_chunk_rate = max(m.upcoming_quality_bitrates[0])
	else:
		next_max_chunk_rate = max_chunk_rate

	incoming_rate_penalty = next_max_chunk_rate / next_tp

	if curr_buf - incoming_rate_penalty >= 0.875 * curr_buf:
		rate_next = chunk_plus
	elif curr_buf <= resv_size:
		rate_next = min_rate
	elif curr_buf >= resv_size + cush_size:
		rate_next = max_rate
	elif rmap >= chunk_plus:
		rate_next = maxLesser(vid_rates,rmap)
	elif rmap <= chunk_minus:
		rate_next = minGreater(vid_rates,rmap)
	else:
		rate_next = rate_prev
	
	# TCP slow start period
	#print(f'current buffer: {curr_buf} incoming: {incoming_rate_penalty} update: {curr_buf - incoming_rate_penalty >= 0.875 * curr_buf}')
	
	# print(f'current buffer: {curr_buf}')
	# print(f'reservoir size: {resv_size}')
	# print(f'cushion size: {cush_size}')
	# print(f'F(b): {rmap}')
	# print(f'chosen rate: {rate_next}\n')

	
	return rate_next

def calcResv(m):
	max_buf = m.buffer_max_size

	if len(m.upcoming_quality_bitrates) > 0:
		next_rate = max(m.upcoming_quality_bitrates[0])
	else:
		next_rate = max(m.quality_bitrates)
	
	# could be improved
	tp = m.previous_throughput

	# limit max reservoir size to half of the buffer
	resv_size = min(next_rate / tp,max_buf / 2)

	upperThresh = max_buf * 0.9

	mapping_range = (resv_size,upperThresh)

	return mapping_range



	


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
	
	# returns a tuple (reservoir size,upper reservoir threshold)
	map_range = calcResv(c)

	# run BBA-2
	res = BBA2(past_qual,c.buffer_seconds_until_empty,map_range[0],map_range[1]-map_range[0],map_range,c.buffer_max_size,c)


	# find index of optimal bitrate
	idx = res

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
	# 	fig,ax = plt.subplots()
	# 	ax.plot(range(len(idxs)),idxs,linewidth=2.0)
	# 	ax.plot(range(len(idxs)),prev_tps,linewidth=2.0)
	# 	ax.plot(range(len(idxs)),bsec,linewidth=2.0)
	# 	ax.set(xlim=(0, 239),
	# 	ylim=(0, 3), yticks=range(1, 30))
	# 	plt.show()
	# 	with open('res1','a') as r:
	# 		r.write(str(sum(bsec)/len(bsec))+'\n')



	return idx  # return BBA-2 selected bitrate