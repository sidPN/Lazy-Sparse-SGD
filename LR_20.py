import math
import sys
import re
import operator

def sigmoid(score):
	overflow = 20.0
	if score > overflow:
		score = overflow
	elif score < -overflow:
		score = -overflow
	exp = math.exp(score)
	return exp/(1 + exp)

def tokenizeDoc(cur_doc):
    return re.findall('\\w+', cur_doc)

def predict(B, V):
	BV = [B[key]*V[key] for key in V.keys()]
	return sigmoid(sum(BV))

def sgd(V, labelPresent):
	global lcl
	global B0
	global A0
	global B1
	global A1
	global B2
	global A2
	global B3
	global A3
	global B4
	global A4
	#-------Initial Predictions-------
	p0 = predict(B0, V)
	p1 = predict(B1, V)
	p2 = predict(B2, V)
	p3 = predict(B3, V)
	p4 = predict(B4, V)
	for id in V:
		if V[id] > 0:
			if B0[id] >= 0.5:
				B0[id] *= math.pow((1 - (2 * lR * reguCoef)), k - A0[id])
			if B1[id] >= 0.5:
				B1[id] *= math.pow((1 - (2 * lR * reguCoef)), k - A1[id])
			if B2[id] >= 0.5:
				B2[id] *= math.pow((1 - (2 * lR * reguCoef)), k - A2[id])
			if B3[id] >= 0.5:
				B3[id] *= math.pow((1 - (2 * lR * reguCoef)), k - A3[id])
			if B4[id] >= 0.5:
				B4[id] *= math.pow((1 - (2 * lR * reguCoef)), k - A4[id])
			B0[id] += (lR * (labelPresent[0] - p0) * V[id])
			B1[id] += (lR * (labelPresent[1] - p1) * V[id])
			B2[id] += (lR * (labelPresent[2] - p2) * V[id])
			B3[id] += (lR * (labelPresent[3] - p3) * V[id])
			B4[id] += (lR * (labelPresent[4] - p4) * V[id])
			A0[id] = k
			A1[id] = k
			A2[id] = k
			A3[id] = k
			A4[id] = k
		# print(id)



vocab = int(sys.argv[1])
initR = float(sys.argv[2])
reguCoef = float(sys.argv[3])
maxIter = int(sys.argv[4])
trainSize = int(sys.argv[5])

#testFile = (sys.argv[6])

currIter = 1
B0 = [0] * vocab #Weight Vector
A0 = [0] * vocab #LastUpdate Vector
B1 = [0] * vocab #Weight Vector
A1 = [0] * vocab #LastUpdate Vector
B2 = [0] * vocab #Weight Vector
A2 = [0] * vocab #LastUpdate Vector
B3 = [0] * vocab #Weight Vector
A3 = [0] * vocab #LastUpdate Vector
B4 = [0] * vocab #Weight Vector
A4 = [0] * vocab #LastUpdate Vector
k = 0
sampleCount = 0
labelDict = {"Person":0, "Place":1, "Species":2, "Work":3, "other":4}
lR = initR/float(currIter * currIter)
LCL = 0
lcl = 0
for line in sys.stdin:
	if sampleCount == trainSize:
		#-------For each parameter, whether it is present in the document or not-------
		LCL += lcl
		currIter += 1
		lR = initR/float(currIter * currIter)
		lcl = 0
		sampleCount = 0
	contents = re.split(r'\t', line)
	labelPresent = [0,0,0,0,0]
	labels = contents[1].split(",")
	for label in labels:
		labelIdx = labelDict[label]
		labelPresent[labelIdx] = 1
	document = tokenizeDoc(contents[2])
	V = {}
	for word in document:
		id = hash(word)%vocab
		if id < 0:
			id += vocab
		if id in V:
			V[id] += 1
		else:
			V[id] = 1
	k += 1
	sgd(V, labelPresent)
	sampleCount += 1

for id in V:
	B0[id] = B0[id] * math.pow((1 - (2 * lR * reguCoef)), k - A0[id])
	B1[id] = B1[id] * math.pow((1 - (2 * lR * reguCoef)), k - A1[id])
	B2[id] = B2[id] * math.pow((1 - (2 * lR * reguCoef)), k - A2[id])
	B3[id] = B3[id] * math.pow((1 - (2 * lR * reguCoef)), k - A3[id])
	B4[id] = B4[id] * math.pow((1 - (2 * lR * reguCoef)), k - A4[id])



# avgLCL = LCL/float(maxIter)
# print(avgLCL)

# Needed for accuracy calculation
correctCount = 0
totalCount = 0
with open(sys.argv[6]) as testFile:
	for line in testFile:
		contents = re.split(r'\t', line)
		labels = contents[1].split(",")
		labelPresent = [0,0,0,0,0]
		for label in labels:
			labelIdx = labelDict[label]
			labelPresent[labelIdx] = 1
		document = tokenizeDoc(contents[2])
		V = {}
		totalCount += 1
		for word in document:
			id = hash(word)%vocab
			if id < 0:
				id += vocab
			if id in V:
				V[id] += 1
			else:
				V[id] = 1
		for label in labelDict:
			if labelDict[label] == 0:
				p0 = predict(B0, V)
				label0 = label
				continue
			if labelDict[label] == 1:
				p1 = predict(B1, V)
				label1 = label
				continue
			if labelDict[label] == 2:
				p2 = predict(B2, V)
				label2 = label
				continue
			if labelDict[label] == 3:
				p3 = predict(B3, V)
				label3 = label
				continue
			if labelDict[label] == 4:
				p4 = predict(B4, V)
				label4 = label
				continue

		print("%s\t%.2f,%s\t%.2f,%s\t%.2f,%s\t%.2f,%s\t%.2f"%(label0, p0, label1, p1, label2, p2, label3, p3, label4, p4))
		for i in range(len(labelDict)):
			if i == 0:
				if (p0 > 0.5 and labelPresent[0] == 1) or (p0 < 0.5 and labelPresent[0] == 0):
					correctCount += 1
			if i == 1:
				if (p1 > 0.5 and labelPresent[1] == 1) or (p1 < 0.5 and labelPresent[1] == 0):
					correctCount += 1
			if i == 2:
				if (p2 > 0.5 and labelPresent[2] == 1) or (p2 < 0.5 and labelPresent[2] == 0):
					correctCount += 1
			if i == 3:
				if (p3 > 0.5 and labelPresent[3] == 1) or (p3 < 0.5 and labelPresent[3] == 0):
					correctCount += 1
			if i == 4:
				if (p4 > 0.5 and labelPresent[4] == 1) or (p4 < 0.5 and labelPresent[4] == 0):
					correctCount += 1
accuracy = correctCount / float(totalCount * 5)
print("Accuracy = %0.9f"%accuracy)
