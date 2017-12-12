##############################################################
#
#
#
#	  This script was made by Shaun Gupta
#
#
#############################################################

from ROOT import *
import sys
import os
sys.argv.append( '-b-' )
sys.argv.append( '-Q-' )
import math
from prettyPalette import *
from array import array
#from treeAnalyserFunctions import *

ROOT.gROOT.LoadMacro("AtlasStyle.C") 
ROOT.gROOT.LoadMacro("AtlasUtils.C") 
ROOT.gROOT.LoadMacro("AtlasLabels.C") 



SetAtlasStyle()
b_m = 0.00001
padScaling = 3./5.  
#padScaling = 6./9.  
padSize = 1-padScaling
BottomScaling=1./(1.-padScaling)
TopScaling=1./padScaling
colours = [1, 4, 2, 3, 11, 8, 9, 46, 30, 7, 29, 28, 33,38, 42, 25, 40]
markers = [8, 24, 25, 26, 27, 28, 30, 32, 4,32, 31, 2,5, 6, 33, 34, 21]	

def convertTGraphToTGraphErrors(graph, errorsX=[], errorsY=[]):

	errors = TGraphErrors()
	ipoint=0
	for i in xrange(graph.GetN()):
		if not graph.GetX()[i]:continue
		print i,graph.GetX()[i]
		errors.SetPoint(i, graph.GetX()[i], graph.GetY()[i])
		errorX = 0.000001*graph.GetX()[i] if not errorsX else errorsX[i]
		errorY = 0.000001*graph.GetY()[i] if not errorsY else errorsY[i]
		errors.SetPointError(i, errorX, errorY)
		ipoint+=1
	errors.SetName(graph.GetName())
	return errors


def createMeanGraph(hist, name, bins, bin_width=1, mean_y=True, x="", y=""):
	new_graph = TGraphErrors()
	new_graph.SetName(name)

	for i in xrange(len(bins)-1):
		if mean_y: projection = hist.ProjectionY("h_projection", (bins[i]/bin_width)+1, bins[i+1]/bin_width)
		else: projection = hist.ProjectionX("h_projection", (bins[i]/bin_width)+1, bins[i+1]/bin_width)
		new_graph.SetPoint(i,(bins[i]+bins[i+1])/2., projection.GetMean()) 
		new_graph.SetPointError(i,(bins[i+1]-bins[i])/2., projection.GetMeanError())
	setTitles(new_graph, x, y)
	return new_graph 

def createEntriesHist(hist_name, resp_distributions, x_bins, y_bins, effective=False):
	hist = TH2D(hist_name, hist_name, len(x_bins)-1, array('d', x_bins), len(y_bins)-1, array('d', y_bins))
	for resp in resp_distributions:
		hist_name = resp.GetName()
		split_name = hist_name.split("_")	
		x_bin = int(split_name[split_name.index("Etruth")+1])-4
		y_bin = int(split_name[split_name.index("x")+1])-4
		hist.SetBinContent(x_bin, y_bin, resp.GetEffectiveEntries() if effective else resp.GetEntries())

	return hist

def Get_error(i, data, debug): 
	if ( data.GetBinContent(i) and debug.GetBinContent(i) ):
		return calc_error(data.GetBinContent(i), debug.GetBinContent(i), data.GetBinError(i), debug.GetBinError(i))
	return 0.0

	
def calc_error(data, debug, data_error, debug_error, mult=0, correlated=False):
        if not data or not debug: return 0
	if mult:
		ratio = debug * data
	else:
		ratio = debug / data
	ferr_data = ( data_error**2 / data**2 ) if data**2 else 0
	ferr_debug = ( debug_error**2 / debug**2 ) if debug**2 else 0
	sqRt = sqrt( ferr_data + ferr_debug ) if not correlated else fabs((debug_error/debug) - (data_error/data))
	return ratio * sqRt 


def getValueAndError(value1, value2, error1, error2, asym=False, quad=False, fracErrors=False, correlated=False, diff=False):
	if quad:
		ratio = TMath.Sqrt(fabs(value2**2-value1**2))
		#ratio = TMath.Sqrt(fabs(value1**2-value2**2))
		if value1 > value2:
			ratio*=-1
		sqRt = TMath.Sqrt(((value1**2)*(error1**2))  + ((value2**2) * (error2**2))) if not correlated else TMath.Sqrt(fabs(((value1**2)*(error1**2))  + ((value2**2) * (error2**2)) - 2*value1*value2*error1*error2))
		error = 0 if not ratio else fabs((1./ratio)) * sqRt
	elif diff:
		ratio = value2 -  value1
		error = TMath.Sqrt(error1**2  + error2**2) if not correlated else TMath.Sqrt(fabs(error2**2 - error1**2))
	elif asym:
                ratio = (value2+2) / (value1+2)
 	        error = calc_error(value1+2, value2+2, error1, error2, correlated=correlated) 
	else:
                ratio = 0 if not value1 else value2 / value1
		if fracErrors:
			error = error2/value2 
		else:
       	                error = calc_error(value1, value2, error1, error2, correlated=correlated) 
#	print error
	return [ratio,error]

def createRatioHist(hist1, hist2, asym=False, diff=False, quad=False, fracErrors=False, skipLimit=False, correlated=False):
	h_ratio = hist1.ProjectionX() if isinstance(hist1, TProfile) else hist1.Clone(hist2.GetName()+"_CLONE_ratio") 
	h_ratio.Reset() 

        ratio =""
        error = 0
        min = 99999
	max = 0 
	arrows=[]
	nEntries=0
        for ii in range(h_ratio.GetNbinsX()+1):
                if not ii or not hist1.GetBinContent(ii):
                        continue
		ratio,error = getValueAndError(hist1.GetBinContent(ii), hist2.GetBinContent(ii), hist1.GetBinError(ii), hist2.GetBinError(ii), asym=asym, fracErrors=fracErrors, quad=quad, diff=diff, correlated=correlated)

		if skipLimit and ratio > 3:
			arrows.append(h_ratio.GetBinCenter(ii))
			continue
		elif not skipLimit and ratio>3:
			arrows.append(h_ratio.GetBinCenter(ii))
			continue
		
                h_ratio.SetBinContent(ii,ratio)
                h_ratio.SetBinError(ii,error)
		nEntries+=1
                if ratio!="" and ((not asym and not quad and ratio) or asym or quad):
			if ratio and (error + ratio)/ratio < 1.5:
				if ratio-error < min: min=ratio-error
				if ratio+error > max: max=ratio+error
			else:
			 	if ratio < min:  min=ratio
				if ratio>max: max = ratio


	return [h_ratio, min, max, arrows, not bool(nEntries)] 


def drawMarkers(x, y, labels, colours, markers, step=0.052, scaling=1, alternate=False, alternate2=False, skipped=[]):
	i=0
	colourI=0
	while i < len(labels):
		if (alternate and i%2) or (alternate2 and (i+1)%2) or (i in skipped):
			i+=1
			continue
		if isinstance(labels[i], dict):
			y,colourI = drawSubLegend(x, y, labels[i], colours, markers, current_colour=colourI, scaling=scaling, skipped=skipped, step=step, current_index=i)
	 	else:
			myMarkerText(x, y, colours[colourI], markers[colourI], labels[i],1, scaling)
			y-=step
			colourI+=1
		i+=1

def drawSubLegend(x, y, label_dict, colours, markers, current_colour=0, current_index=0, step=0.052, scaling=1, skipped=[]):
	y-=step*0.3
	title = label_dict.keys()[0]
	labels = label_dict.values()[0]
	myText(x-0.01, y, 1, title, scaling)
	y-=step*0.7
	#y-=step*0.6
	i=0
	while i < len(labels):
		if current_index in skipped: 
			i+=1
			current_index+=1
			continue
		myMarkerText(x, y, colours[current_colour], markers[current_colour], labels[i],1, scaling)
#		myMarkerText(x+0.02, y, colours[current_colour], markers[current_colour], labels[i],1, scaling*0.8)
		y-=step
		#y-=step*0.8	
		i+=1
		current_index+=1
		current_colour+=1		
	#y-=step*0.2
	return [y,current_colour]


def drawLabels(labels, x, y, step=0.052, scaling=1):
	i=0
	while i < len(labels):
		if not labels[i]:
			i+=1
			continue
		if "0 <= |#eta" in labels[i]:
			labels[i] = labels[i][labels[i].find("|"):]
       		#myText(x, y, 1, labels[i], 0.03)
       	 	myText(x, y, 1, labels[i], scaling)
		i+=1
		y-=step


def getGraphMaxX(graphs):
	maxX = -1
	maxi = -1
	for i in xrange(len(graphs)):
		if graphs[i].GetN() and maxX < graphs[i].GetX()[graphs[i].GetN()-1]:
			maxX=graphs[i].GetX()[graphs[i].GetN()-1]
			maxi=i

	return maxi

def createRatioGraph(graph1, graph2, bins=[], asym=False, diff=False, quad=False, fracErrors=False, skipLimit=False, noRatioLimit=False, correlated=False):
	graphValues = []

	arrows=[]
#	graphValues.append([graph1.GetX(), graph1.GetY(), graph1.GetEX(), graph1.GetEY()])
#	graphValues.append([graph2.GetX(), graph2.GetY(), graph2.GetEX(), graph2.GetEY()])
#	print graph1.GetX()
	ratioValues = [[], [], [], []]
	
#	curX = getAxisTitle(graph1.GetName().split("_")[1])
#        h_ratio.Divide(graph1)
        min = 999999999
	max= 0
	i1=0
	i2=0
	debug=False
#	if "g_rms_rel_mean_eta_0" in graph1.GetName():
#		debug=True
	if debug: 
		print "START"
		print "CORRELATED:\t",correlated
		print "QUADRATURE?",quad
		print "DIFFERENCE?",diff
	h_ratio = graph1.Clone(graph2.GetName() + "_ratio")
	h_ratio.Set(0)
	while i1!= graph1.GetN() and i2!= graph2.GetN():
		if not graph1.GetY()[i1] and not graph2.GetY()[i2]:
			i1+=1
			i2+=1
			continue
		elif not graph1.GetY()[i1]:
			i1+=1
			continue
		elif not graph2.GetY()[i2]:
			i2+=1
			continue
		elif bins and findBin(bins, graph2.GetX()[i2]) < findBin(bins, graph1.GetX()[i1]):
			i2 += 1
			continue
		elif bins and findBin(bins, graph2.GetX()[i2]) > findBin(bins, graph1.GetX()[i1]):
			i1 += 1
			continue

		ratio,error = getValueAndError(graph1.GetY()[i1], graph2.GetY()[i2], graph1.GetEY()[i1], graph2.GetEY()[i2], asym=asym, fracErrors=fracErrors, quad=quad, diff=diff, correlated=correlated)
		if debug: 
			print "INDICIES:",i1,i2
			print "Ratio:",ratio,"Error:",error,"Uncorrelated Error:",getValueAndError(graph1.GetY()[i1], graph2.GetY()[i2], graph1.GetEY()[i1], graph2.GetEY()[i2], asym=asym, fracErrors=fracErrors, quad=quad, diff=diff, correlated=False)[1]
			print "Value1:",graph1.GetY()[i1],"Error1:",graph1.GetEY()[i1],"Value2:",graph2.GetY()[i2],"Error2:",graph2.GetEY()[i2]

##		ratio = graphValues[1][1][ii] / graphValues[0][1][ii]

#                #error = calc_error(graphValues[1][1][ii], graphValues[0][1][ii], graphValues[1][3][ii], graphValues[0][3][ii])

#		ratioValues[0].append(ratio)
##		ratioValues[1].append(error)
#		ratioValues[1].append(0.0000001)
#		ratioValues[2].append(graph1.GetX()[i1])
#		ratioValues[3].append(graph1.GetEX()[i1])
		if skipLimit and ratio > 3: 
			i2+=1
			i1+=1
			arrows.append(graph1.GetX()[i1])
			continue
		elif not skipLimit and ratio>3:
			i2+=1
			i1+=1
			arrows.append(graph1.GetX()[i1])
			continue

		h_ratio.SetPoint(h_ratio.GetN(), graph1.GetX()[i1], ratio)
		h_ratio.SetPointError(h_ratio.GetN()-1, graph1.GetEX()[i1], error)

                if (not asym and not quad and ratio) or asym or quad:
			if ratio and (error+ratio)/ratio < 1.5:
				if ratio-error < min: min=ratio-error
				if ratio+error > max: max=ratio+error
			else:
			 	if ratio < min:  min=ratio
				if ratio>max: max = ratio		
		i2+=1
		i1+=1
##		ratioValues[0].append(


#	if ratioValues[2]:
#		h_ratio = TGraphErrors(len(ratioValues[2]), array('d',ratioValues[2]), array('d', ratioValues[0]), array('d', ratioValues[3]), array('d', ratioValues[1])) 
#	else:
#		h_ratio = TGraphErrors()
#       	gPad.SetPad(0,0,1.0,0.3)

	if debug: print "END"

	return [h_ratio, min, max, arrows, not bool(h_ratio.GetN())]






def legendOrganiser(legend):

	modified_legend = []
	bin_labels = []
	for label in legend:
		if "#int" in label:
			modified_legend.insert(0, label)
		elif "<" in label or ">" in label or "#le" in label:
			bin_labels.append(label)
		else:
			modified_legend.append(label)

	modified_legend.extend(bin_labels)	

	return modified_legend

def createTH2D(hists1D, xbins, ybins, histName, absolute=False):

	if isinstance(hists1D[0], TH1):
		return createTH2DHist(hists1D, xbins, ybins, histName)
	else:
		return createTH2DGraph(hists1D, xbins, ybins, histName, absolute)


def createTH2DGraph(graphs1D, xbins, ybins, histName, absolute=False):
	set_palette()
	hist2D = TH2D(histName, histName, len(xbins)-1, array('d', xbins), len(ybins)-1, array('d', ybins))

	for ibinx in xrange(hist2D.GetNbinsX()):
		#for ibiny in xrange(hist2D.GetNbinsY()):
		for ibiny in xrange(graphs1D[ibinx].GetN()):
			y_bin = findBin(ybins, graphs1D[ibinx].GetX()[ibiny])
			if absolute:
				hist2D.SetBinContent(ibinx+1, y_bin+1, math.fabs(graphs1D[ibinx].GetY()[ibiny]))
			else:
				hist2D.SetBinContent(ibinx+1, y_bin+1, graphs1D[ibinx].GetY()[ibiny])
			hist2D.SetBinError(ibinx+1, y_bin+1, graphs1D[ibinx].GetEY()[ibiny])

	return hist2D
	
def createTH2DFromTGraph2D(graph2D, xbins, ybins, histName, verbose=False):
	set_palette()
	hist2D = TH2D(histName, histName, len(xbins)-1, array('d', xbins), len(ybins)-1, array('d', ybins))
	if verbose:
		print "DEBUG"
		print graph2D.GetName()
	shifted_graphs = []

	for i in xrange(graph2D.GetN()):
		graph_shifted = graph2D.Clone(graph2D.GetName() + "_" + str(i))
		graph_shifted.SetPoint(i, graph2D.GetX()[i], graph2D.GetY()[i], graph2D.GetZ()[i]+graph2D.GetEZ()[i])
		shifted_graphs.append(graph_shifted)

	for ibinx in xrange(hist2D.GetNbinsX()):
		for ibiny in xrange(hist2D.GetNbinsY()):
			x = hist2D.GetXaxis().GetBinCenter(ibinx+1)
			y = hist2D.GetYaxis().GetBinCenter(ibiny+1) 
			nominal = graph2D.Interpolate(x, y)
			if verbose:
				print "x:%.2f\ty:%.2f\tnominal:%.2f"%(x, y, nominal)
			hist2D.SetBinContent(ibinx+1, ibiny+1, nominal)
			error = 0
			for graph in shifted_graphs:
				error += ((nominal - graph.Interpolate(x, y))**2)
			hist2D.SetBinError(ibinx+1, ibiny+1, sqrt(error))


	return hist2D


def createTH2DHist(hists1D, xbins, ybins, histName):
	set_palette()
	hist2D = TH2D(histName, histName, len(xbins)-1, array('d', xbins), len(ybins)-1, array('d', ybins))

	for ibinx in xrange(hist2D.GetNbinsX()):
		hist = hists1D[ibinx]
		for ibiny in xrange(hist2D.GetNbinsY()):
			y_bin_center = (ybins[ibiny] + ybins[ibiny])/2.
			current_bin = hists1D[ibinx].FindBin(y_bin_center)
			hist2D.SetBinContent(ibinx+1, ibiny+1, hists1D[ibinx].GetBinContent(current_bin))
			hist2D.SetBinError(ibinx+1, ibiny+1, hists1D[ibinx].GetBinError(current_bin))

	return hist2D
			

def createScaleFactorHist(histGood, histToScale):

	scaleFactors = histToScale.Clone("scaleFactorHist")
	histGoodNormalized = histGood.Scale(1./histGood.Integral())
	histToScaleNormalized = histToScale.Scale(1./histToScale.Integral())
	for ibin in xrange(histGood.GetNbinsX()):
		scaleFactors.SetBinContent(ibin+1, histGoodNormalized.GetBinContent(ibin+1)/histToScaleNormalized.GetBinContent(ibin+1))
		scaleFactors.SetBinError(ibin+1, Get_error(ibin+1, histGoodNormalized, histToScaleNormalized))

	return scaleFactors

def createTGraph2D(graphsX, graphsY, xbins, ybins, verbose=False):
	#Note: This function plots absolute of values on Z axis
	set_palette()
	if verbose:
		print "DEBUG"

	if not graphsX:
		print "SHIT"
		return TGraph2DErrors()

	graph2D = TGraph2DErrors()
	graph2D.SetName(graphsX[0].GetName() + "_graph2D")
#	graph2D.GetXaxis().SetName(xVar)
#	graph2D.GetYaxis().SetName(yVar)
#	graph2D.GetZaxis().SetName(zVar)
	
	npoints =0 
	for i in xrange(len(graphsY)):
		if not graphsY[i].GetN():
			continue
		for ipoint in xrange(graphsY[i].GetN()):
			curYBin = findBin(ybins, graphsY[i].GetX()[ipoint])
#			print ybins
#			print graphsY[i].GetX()[ipoint]
#			print graphsY[i].GetN()
#			print graphsX[curYBin].GetN()
#			print i
			if not graphsX[curYBin].GetN():
				if verbose:		
					print "CRAP"
				continue
			curXbin=0
			xValue=0
			xValueE=0
			for point in xrange(graphsX[curYBin].GetN()):
				curXbin = findBin(xbins, graphsX[curYBin].GetX()[point])
				if curXbin == i:
					xValue = graphsX[curYBin].GetX()[point]
					xValueE = graphsX[curYBin].GetEX()[point]
					break
			if not xValue:
				if verbose:		
					print "CRAP2"
				continue
			graph2D.SetPoint(npoints, xValue, graphsY[i].GetX()[ipoint], math.fabs(graphsY[i].GetY()[ipoint]))
			graph2D.SetPointError(npoints, xValueE, graphsY[i].GetEX()[ipoint], graphsY[i].GetEY()[ipoint])
			npoints+=1

	return graph2D

def createNonClosureGraph(graphCorrected):
	
	graph_non_closure = graphCorrected.Clone(graphCorrected.GetName() + "_non_closure")
	for i in xrange(graph_non_closure.GetN()):
		graph_non_closure.SetPoint(i, graph_non_closure.GetX()[i], graphCorrected.GetY()[0]-graphCorrected.GetY()[i]) #Using 0 segment bin as baseline for non-closure..
		#graph_non_closure.SetPointError(i, graph_non_closure.GetEX()[i], calc_error(graphCorrected.GetY()[i],graphCorrected.GetY()[i],graphCorrected.GetEY()[i], graphCorrected.GetEY()[i]))
	graph_non_closure.GetYaxis().SetTitle("Non Closure")
	return graph_non_closure



def graphConvertAsymmetryToResponse(graphAsymmetry):
	
	graph_response = graphAsymmetry.Clone(graphAsymmetry.GetName() + "_response")
	for i in xrange(graph_response.GetN()):
		graph_response.SetPoint(i, graph_response.GetX()[i], (1+(graphAsymmetry.GetY()[i]/2.))/(1-(graphAsymmetry.GetY()[i]/2.)))
		graph_response.SetPointError(i, graph_response.GetEX()[i],  graph_response.GetEY()[i]/((1-(graphAsymmetry.GetY()[i]/2.))**2))

	return graph_response


def graphSubtraction(graphNom, graphSub, xBinning, asymmetry=False):
	
	graph_subtracted = graphNom.Clone(graphNom.GetName() + "_subtracted")
	i1=0
	i2=0
	while i1!= graph_subtracted.GetN() and i2!= graphSub.GetN():
		if findBin(xBinning, graphSub.GetX()[i2]) < findBin(xBinning, graph_subtracted.GetX()[i1]):
			i2 += 1
			continue
		elif findBin(xBinning, graphSub.GetX()[i2]) > findBin(xBinning, graph_subtracted.GetX()[i1]):
			i1 += 1
			continue
		if asymmetry:
			graph_subtracted.SetPoint(i1, graph_subtracted.GetX()[i1], graphNom.GetY()[i1] - graphSub.GetY()[i2])
		else:
			graph_subtracted.SetPoint(i1, graph_subtracted.GetX()[i1], graphNom.GetY()[i1] - (1-graphSub.GetY()[i2]))
		
		graph_subtracted.SetPointError(i1, graph_subtracted.GetEX()[i1], sqrt(graphNom.GetEY()[i1]**2 + graphSub.GetEY()[i2]**2))
		i1+=1
		i2+=1
	return graph_subtracted

def convertMultiTGraphToHist(graphs):
	hists = []
	for graph in graphs:
		hists.append(convertTGraphToHist(graph))
	return hists

def convertTGraphToHist(graph):
	print graph
	if not graph.GetN():
		return TH1F()
	print graph.GetN()
	bins = []
	bins.append(graph.GetX()[0]-graph.GetEX()[0])

	for i in xrange(graph.GetN()):
		bins.append(graph.GetX()[i]+graph.GetEX()[i])

	hist = TH1F(graph.GetName()+"_hist", graph.GetName()+"_hist", graph.GetN(), array('d', bins))
	
	for i in xrange(graph.GetN()):
		hist.SetBinContent(i+1, graph.GetY()[i])
		hist.SetBinError(i+1, graph.GetEY()[i])
	hist.GetXaxis().SetName(graph.GetXaxis().GetName())
	hist.GetYaxis().SetName(graph.GetYaxis().GetName())

	return hist


def dynamical_binning(h_x, n): 
  totalEntries = 0
  for j in xrange(1, h_x.GetNbinsX()+1):
    totalEntries += h_x.GetBinContent(j)
  
  #std::cout<<"totalEntries  : "<<totalEntries<<std::endl;

  lowBin=[]
  highBin=[]
  entriesN=[]

  for i in xrange(n):
    lowBin.append(0)
    highBin.append(0)
    entriesN.append(0)

  lowBin[0] = 0
  firstBin = -1
  lastBin = -1
  for k in xrange(0, n):
    entries = 0

    for j in xrange(1, h_x.GetNbinsX()+1):

      if(firstBin != -1 and h_x.GetBinContent(j) == 0):
	lastBin = j-1
      
      
      entries += h_x.GetBinContent(j)

      if(firstBin == -1 and entries > 0):
	firstBin = j
      
      if (entries >= (k+1.)*(1.0*totalEntries)/(1.0*n)):


	if(k<n-1): lowBin[k+1] = j+1
	highBin[k] = j
	if(highBin[k]<lowBin[k]): lowBin[k] = highBin[k]
	entriesN[k] = entries
	break
      
    
  
  
  lowBin [0]   = firstBin
  highBin[n-1] = lastBin
  if(lastBin < lowBin[n-1]): highBin[n-1] = h_x.GetNbinsX()
  
  return [highBin, lowBin, entriesN]

def getMaximumYGraphs(graphs, isGraph=False, skipped=[]):
	maxY = 0
	for i in xrange(len(graphs)):
		if i in skipped: continue
		graph = graphs[i]

		curMax = getMaximumYGraph(graph) if isGraph else  getMaximumYHist(graph)
		if curMax > maxY:
			maxY = curMax
	return curMax

def getMaximumYGraph(graph):
	maxY =0
	for i in xrange(graph.GetN()):
		if not i or (graph.GetY()[i]+graph.GetEY()[i]) > maxY:
			maxY = graph.GetY()[i]+graph.GetEY()[i]
	return maxY

def getMaximumYHist(graph):
	maxY = 0 
	for i in xrange(graph.GetXaxis().GetNbins()+1):
		if not i: continue
		if i==1 or (graph.GetBinContent(i)+graph.GetBinError(i)) > maxY:
			maxY = graph.GetBinContent(i)+graph.GetBinError(i)

	return maxY


def getMinimumYGraphs(graphs, isGraph=False, log=False, skipped=[]):
	minY = 99999
	for i in xrange(len(graphs)):
		if i in skipped: continue
		graph = graphs[i]
		curMin = getMinimumYGraph(graph, log=log) if isGraph else  getMinimumYHist(graph, log=log)
		if curMin < minY:
			minY = curMin
	return minY 

def getMinimumYGraph(graph, log=False):
	minY = 99999
	for i in xrange(graph.GetN()):
		if (graph.GetY()[i]-graph.GetEY()[i]) < minY and ((not log) or (log and ((graph.GetY()[i]-graph.GetEY()[i])>0))):
			minY = graph.GetY()[i]-graph.GetEY()[i]

	return minY

def getMinimumYHist(graph, log=False):
	minY = 99999
	for i in xrange(graph.GetXaxis().GetNbins()+1):
		if not i: continue
		if graph.GetBinContent(i)-graph.GetBinError(i) < minY and ((not log) or (log and ((graph.GetBinContent(i)-graph.GetBinError(i))>0))):
			minY = graph.GetBinContent(i)-graph.GetBinError(i)

	return minY

def getMinimumXMulti(hists, graph=False, skipped=[]):
	minX = 1000
	for i in xrange(len(hists)):
		if i in skipped: continue
		hist = hists[i]
		if graph and hist.GetN():
			curMinX = hist.GetX()[0] - hist.GetEX()[0]
			print curMinX
		elif not graph:
			curMinX = getMinimumX(hist)
		else: curMinX=1000

		if minX > curMinX:
			minX = curMinX

	if minX == 1000:
		return 0 


	#return 0.95*minX if minX > 0 else 1.05*minX
	return minX

def getMinimumX(hist):
	for iBin in xrange(hist.GetNbinsX()+1):
		if iBin  and hist.GetBinContent(iBin):
			return hist.GetXaxis().GetBinLowEdge(iBin)	
	return 1000 

def getMaximumXMulti(hists, graph=False, skipped=[]):
	maxX = 0

	for i in xrange(len(hists)):
		if i in skipped: continue
		hist = hists[i]

		if graph:
			curMaxX = 0 if not hist.GetN() else hist.GetX()[hist.GetN()-1] + hist.GetEX()[hist.GetN()-1]
		else:
			curMaxX = getMaximumX(hist)
		if maxX < curMaxX:
			maxX = curMaxX

	#return maxX*1.1 if maxX > 0 else maxX*0.9
	return maxX

def getMaximumX(hist):
	maxBin = 0
	for iBin in xrange(hist.GetNbinsX()+1):
		if hist.GetBinContent(iBin):
			maxBin=iBin
	return hist.GetXaxis().GetBinUpEdge(maxBin)	

def findBin(bins, value):
	for i in xrange(len(bins)-1):
		if (bins[i] <= value and bins[i+1] > value):
			return i
def createBinLabels(bins, variable):

	labels = []

	for i in xrange(len(bins)-1):
		labels.append(str(bins[i]) + " <= " + variable + "< " + str(bins[i+1]))
	return labels

def getBinnedHists(inFile, bins, histName, xTitle="", yTitle="", muErr=False, pTErr=False, rebin=[]):
	if not bins or "lead" in bins[0]:
		if not bins:
			hist = inFile.Get(histName)
			print histName
			setTitles(hist, xTitle, yTitle)
			if rebin:
				hist.Rebin(len(rebin)-1, hist.GetName(), array('d', rebin))
			return hist
		else:
			hists = []
			hists.append(inFile.Get(histName + "_lead"))
			setTitles(hists[0], xTitle, yTitle)
			hists.append(inFile.Get(histName + "_subl"))
			setTitles(hists[1], xTitle, yTitle)
			if rebin:
				hists[0].Rebin(len(rebin)-1, hists[0].GetName(), array('d', rebin))
				hists[1].Rebin(len(rebin)-1, hists[1].GetName(), array('d', rebin))
			return hists
	curBin = []
	for bin in xrange(bins[0][1]-1):
		name = "%s_%s"
		if (("muon" in bins[0][0]) and muErr) or (("pT" in bins[0][0]) and pTErr):
			name+="%i"
		else:
			name+="_%i"
		curBin.append(getBinnedHists(inFile, bins[1:], name%(histName,bins[0][0],bin), xTitle, yTitle, muErr, pTErr, rebin))

	return curBin


def mergeHists(lead, subl):
	if isinstance(lead, TH1) or isinstance(lead, TGraph):
		lead  = lead.Clone(lead.GetName().replace("lead", "both"))
		lead.Add(subl)
		lead.GetXaxis().SetTitle(lead.GetXaxis().GetTitle().replace("Leading", "Leading/Subleading"))
		lead.GetYaxis().SetTitle(lead.GetYaxis().GetTitle().replace("Leading", "Leading/Subleading"))

		return lead
	else:
		currentBin = []
		N = len(lead)

		for i in xrange(N):
			currentBin.append(mergeHists(lead[i], subl[i]))
		if len(currentBin) == 1:
			return currentBin[0]
		return currentBin


def getProfileHistograms(histograms, profY=False, rebin=[]):

	if isinstance(histograms, TH2):
		if profY:
			profile = histograms.ProfileY()
			profile.GetXaxis().SetTitle("<" + histograms.GetXaxis().GetTitle() + ">")
			profile.GetYaxis().SetTitle(histograms.GetYaxis().GetTitle())
			
		else:
			profile = histograms.ProfileX()
			profile.GetYaxis().SetTitle("<" + histograms.GetYaxis().GetTitle() + ">")
			profile.GetXaxis().SetTitle(histograms.GetXaxis().GetTitle())

		if rebin:
			profile.Rebin(len(rebin)-1, profile.GetName(), array('d', rebin))
		return profile
	else:
		profiles = []
		for histogram in histograms:
			profiles.append(getProfileHistograms(histogram, rebin=[]))
		return profiles 

def makeFlatLegend(legend):
	flat_legend = []
	for label in legend:
		if isinstance(label, dict):
			for l in label.values()[0]:
				flat_legend.append(l)
		else:
			flat_legend.append(label)
	return flat_legend

def convertTListtoList(rList):
	if not rList:
		return []
	outList = []

	next = TIter(rList)
	obj = 1
	
	while obj != rList.Last():
					
		obj = next()
		
		if isinstance(obj, TList):
			outList.append(convertTListtoList(obj))
		else:
			outList.append(obj)

	return outList


def comparePlots(graphs, colours, markers, outFile, labels,  globalLabels=[], logX=False, logY=False, bins=[], bottom=False, right=False, ratio=True, rangeX=[], quad=False, diff=False, asym=False, returnRatio=False, min=-1, max=-1, ATLAS="", xTitle="", yTitle="", ratioYRange=[], resp=False, GSC=[], noRatioLimit=False, alternate=False, alternate2=False, fracErrors=False, leftLegend=False, graph=False, autoXRange=False, extraOpt="", pt_resol=False, correlated=False, LCW=False):
	style = AtlasStyle()
	gROOT.SetStyle("ATLAS")
	style.cd()
	gROOT.ForceStyle()
#	SetAtlasStyle()
	if globalLabels:
		globalLabels = legendOrganiser(globalLabels)
#	correlated=False	
	print graphs	
	#segs = "segments" in xTitle or "segments" in yTitle or "segments" in graphs[0].GetXaxis().GetTitle() or "segments" in graphs[0].GetXaxis().GetTitle()
	if ratio:
	   	#canv = TCanvas("c0", "c0", 690, 650)
	   	#canv = TCanvas("c0", "c0", 750, 650)
	        canv = TCanvas("c0", "c0", 700, 650)
		canv.UseCurrentStyle()
		pad0 = TPad(graphs[0].GetName() +  "plot", graphs[0].GetName() +"plot", 0., padSize, 1.,              1.)
		pad0.Draw()
		pad1 = TPad(graphs[0].GetName() +"ratio",graphs[0].GetName() +"ratio", 0.,              0., 1., padSize) 
		pad1.Draw()
        	pad1.SetTopMargin(0.0)
	       	pad1.SetBottomMargin(style.GetPadBottomMargin()*(BottomScaling))
		pad0.cd()
       		pad0.SetBottomMargin(0)
       		pad0.SetTopMargin(style.GetPadTopMargin())
       		#pad0.SetLeftMargin(0.19)
       		#pad1.SetLeftMargin(0.19)
		if pt_resol: pad0.SetLeftMargin(pad0.GetLeftMargin()*1.25)
       		pad0.SetRightMargin(style.GetPadRightMargin())
       		pad1.SetRightMargin(style.GetPadRightMargin())
       		#pad0.SetBottomMargin(b_m)
       		#pad0.SetBottomMargin(b_m)


	else:
	        canv = TCanvas("c0", "c0", 700, 500)
		canv.UseCurrentStyle()
       		canv.cd(0)


	isGraph = isinstance(graphs[0], TGraph)

	if isGraph:
		m = TMultiGraph()
		draw_opt="AP%s"%extraOpt
	else:
		m = THStack("new", "new")
		draw_opt="nostackE%s"%extraOpt
	skipped = []
	colourI=0
	flat_legend = makeFlatLegend(labels)
	for i in xrange(len(graphs)):
		graph = graphs[i]
		if (alternate and i%2) or (alternate2 and (i+1)%2) or len(graphs) > 2 and ((isGraph and not graph.GetN()) or  (not isGraph and not graph.GetEntries())) :
		#if (alternate and i%2) or (alternate2 and (i+1)%2) or len(graphs) > 2 and ((isGraph and not graph.GetN()) or  (not isGraph and not graph.GetEntries())) or (segs and "AFII" in flat_legend[i]):
			skipped.append(i)
			continue

	        graph.SetLineColor(colours[colourI])
	        graph.SetMarkerStyle(markers[colourI])
	        graph.SetMarkerSize(style.GetMarkerSize()*TopScaling)
		graph.GetYaxis().SetTitleSize(style.GetTitleYSize()*TopScaling)
		graph.GetYaxis().SetLabelSize(style.GetLabelSize("Y")*TopScaling)
		graph.GetYaxis().SetTitleOffset(0.9)
		
		#graph.GetYaxis().SetTitleOffset(style.GetTitleYOffset())
		graph.GetXaxis().SetTitleSize(style.GetTitleXSize()*TopScaling)
		graph.GetXaxis().SetTitleOffset(style.GetTitleXOffset()*TopScaling)
		graph.GetXaxis().SetLabelSize(style.GetLabelSize("X")*TopScaling)
	        #graph.SetMarkerSize(1.25)
		graph.SetMarkerColor(colours[colourI])
		m.Add(graph.Clone(graph.GetName() + "_clone"))
		colourI+=1	

	if not colourI:
		if returnRatio:
			return TGraphErrors() if isGraph else TH1F()
		return

	m.Draw(draw_opt)

	
	m_hist = m if isGraph else m.GetHistogram()
	m_max = m.GetHistogram() if isGraph else m


	m_hist.GetXaxis().SetTitle(graphs[0].GetXaxis().GetTitle())
	m_hist.GetYaxis().SetTitle(graphs[0].GetYaxis().GetTitle())



	if xTitle:
		m_hist.GetXaxis().SetTitle(xTitle)
	if yTitle:
		m_hist.GetYaxis().SetTitle(yTitle)


	m_hist.GetYaxis().SetTitleSize(graphs[0].GetYaxis().GetTitleSize())
	m_hist.GetYaxis().SetLabelSize(graphs[0].GetYaxis().GetLabelSize())
	m_hist.GetYaxis().SetTitleOffset(graphs[0].GetYaxis().GetTitleOffset())
	m_hist.GetXaxis().SetTitleSize(graphs[0].GetXaxis().GetTitleSize())
	m_hist.GetXaxis().SetLabelSize(graphs[0].GetXaxis().GetLabelSize())
	m_hist.GetXaxis().SetTitleOffset(graphs[0].GetXaxis().GetTitleOffset())

	'''
	if rangeX:
	       	m_hist.GetXaxis().SetRangeUser(rangeX[0], getMaximumXMulti(graphs)) if (rangeX[1] == -1) else m_hist.GetXaxis().SetRangeUser(rangeX[0], rangeX[1])
		"" if not isGraph else m.Draw("AP" + extraOpt)
		gPad.Update()
	'''

	if autoXRange or rangeX:
		#minX = m_hist.GetXaxis().GetXmin() 
		minX = m_hist.GetXaxis().GetXmin() if not isGraph else getMinimumXMulti(graphs, isGraph, skipped=skipped) 
		maxX = getMaximumXMulti(graphs, isGraph, skipped=skipped)
		#print "maxX",maxX
		#if logX: maxX += (maxX-minX)*0.05
		if logX: 
			maxX *= (10**((log10(maxX)-log10(minX))*0.03))
			minX *= (10**((log10(maxX)-log10(minX))*-0.08))
		else:
			maxX = maxX*1.08 if maxX > 0 else maxX*0.92
			minX = 0.95*minX if minX > 0 else 1.05*minX

		minX = minX if not rangeX or rangeX[0]==-1 else rangeX[0]
		maxX = maxX if not rangeX or rangeX[1]==-1 else rangeX[1]
#		print minX
		#sys.exit()
		if isGraph: m_hist.GetXaxis().SetLimits(minX, maxX)
		else: m_hist.GetXaxis().SetRangeUser(minX, maxX)
		"" if not isGraph else m.Draw(draw_opt)
		gPad.Update()

	#m.SetMaximum(minMax[1]*1.05)


#	if "segs" in graphs[0].GetName():
#		m.SetMaximum(m_max.GetMaximum()*2)
#	if "rms_rel_mean" in graphs[0].GetName() and "0" in graphs[0].GetName(): 
#		print "MAX_TEST_FEB\t",max 
#		print "MIN_TEST\t",min

	min = getMinimumYGraphs(graphs,isGraph=isGraph, log=logY, skipped=skipped) if min == -1 else min
	max = getMaximumYGraphs(graphs, isGraph=isGraph, skipped=skipped) if max == -1 else max

	#max = m_max.GetMaximum() if max == -1 else max
	s_factor = 3 if not asym or "resp" in graphs[0].GetName() or "resol" in graphs[0].GetName() else 2
	if not logY:
		#m_max.SetMaximum(m_max.GetMaximum())
		#if "asym" in graphs[0].GetName() or "resp" in graphs[0].GetName(): m_max.SetMinimum(min)
		#else: m_max.SetMinimum(min - ((10**(math.floor(0 if not max else math.log10(abs(max)))-1)) * 2))
		#m_max.SetMinimum(min)
		#m_max.SetMinimum(min - ((10**(math.floor(0 if not max else math.log10(abs(max)))-1)) * s_factor))
		m_max.SetMinimum(min - ((max - min) * 0.1))
		#print 0 if not max else math.log10(max)
		#if "4" in graphs[0].GetName() and not "resp" in graphs[0].GetName():exit()

		m_max.SetMaximum(max + ((max - m_max.GetMinimum())*1.6))

		#m_max.SetMaximum(max + ((max - m_max.GetMinimum())*1.2))

	#	if "11" in graphs[0].GetName() and not "resp" in graphs[0].GetName() and "h_" in graphs[0].GetName():exit()
	else:
#		m_max.SetMaximum(m_max.GetMaximum() + 10**(abs(((log(m_max.GetMaximum()) if m_max.GetMaximum() else 0)  - (log(m_max.GetMinimum()) if m_max.GetMinimum() else 0)))*3.))
		m_max.SetMaximum(max * 10**(abs(((0 if not max else math.log10(max))  - (0 if not min else math.log10(min)))) * 1.3))
		#m_max.SetMaximum(max * 10**(abs(((0 if not max else math.log10(max))  - (0 if not min else math.log10(min)))) * 0.95))
#		if "11" in graphs[0].GetName() and not "resp" in graphs[0].GetName():exit()
		#m_max.SetMinimum(min)
		#m_max.SetMinimum(min - ((10**(math.floor(0 if not max else math.log10(abs(min)))-1)) * 1./s_factor))
		m_max.SetMinimum(min*10**(abs(((0 if not max else math.log10(max))  - (0 if not min else math.log10(min)))) * -0.15))
	
#		if "eta_0" in graphs[0].GetName() and "pT_4" in graphs[0].GetName():
#			print min,max
#			sys.exit()
		#m_max.SetMaximum(m_max.GetYaxis().GetXmax() + ((m_max.GetYaxis().GetXmax() - m_max.GetYaxis().GetXmin())*1./3.))
#	if min!= -1:
#		m_max.SetMinimum(min)
#	if max!=-1:
#		m_max.SetMaximum(max*1.6 if asym else max*1.1)


	

	if bottom or (leftLegend and not right):
		y = 0.5
	else:
		y = 0.85
		#y = 0.88
	i=0
	
	if right:
	#if right or ("segs" in graphs[0].GetName() and not isGraph):
		x = 0.65
		#x2 = 0.22
		x2 = 0.2
	else:
		x = 0.2
		#x = 0.2
		#x = 0.2
		#x2 = 0.62
		x2 = 0.72
		#x2 = 0.7

	if leftLegend and not right:
		x2=0.2
	if pt_resol:
#		x-=0.02
		x2-=0.2
		x+=0.05
	step = 0.063
	#step = 0.059
	#step = 0.058
	#step = 0.052
	#step*=(1.4 if ratio else 1)
	step*=(1.4 if ratio else 1)
	#step*=(1.2 if ratio else 1)
	#step = 0.052*1.2 if ratio else 0.052
	#scaling = 1.2 if ratio else 1
#	scaling = 1.65 if ratio else 1
#	scaling = 1.4 if ratio else 1
	scaling = 1.7 if ratio else 1
	#scaling = 1.53 if ratio else 1
	#scaling = 1.55 if ratio else 1

	if ATLAS:
		ATLASLabel(x,y, ATLAS, 1, scaling)
		y-=step

	drawLabels(globalLabels, x, y, step=step, scaling=scaling)

	legend_step =  0.06*1.2 if ratio else 0.06

		

	if bottom:
	#if bottom and not ("segs" in graphs[0].GetName() and not isGraph):
		y2=0.6
		drawMarkers(x2, y2, labels, [1,4] if len(graphs)<3 else colours, [8,4] if len(graphs)<3 else markers, scaling=scaling, step=step, alternate=alternate, alternate2=alternate2, skipped=skipped)


	else:
		y2=0.88 if not pt_resol and not LCW else 0.58
		#if (leftLegend and not right) or ("segs" in graphs[0].GetName() and not isGraph):
		if (leftLegend and not right):
			y2 = y-(len(globalLabels)*step)
		drawMarkers(x2, y2, labels, [1,4] if len(graphs)<3 else colours, [8,4] if len(graphs)<3 else markers, scaling=scaling, step=step, alternate=alternate, alternate2=alternate2, skipped=skipped)

	gPad.SetLogx(logX)	
	gPad.SetLogy(logY)
	m_hist.GetXaxis().SetMoreLogLabels(logX)

	if asym or resp:
		line2 = TLine(m.GetXaxis().GetXmin(), 0 if asym else 1, m.GetXaxis().GetXmax(), 0 if asym else 1)
		line2.SetLineStyle(2)
		line2.Draw("same")



	if GSC:
	        base = TF1("base","pol0",-5,300)
       	 #base.SetParameter(0, maxY)
		#norm = 0.2
		minY = getMinimumYGraphs(graphs, isGraph=isGraph, log=logY)
       		base.SetParameter(0,min*0.6 if min > 0 else min*1.4) # Need to adjust this...
	#	if minY < 0:
#			norm = minY * 1.2
 	#     	  	base.SetParameter(0, norm)
#		else:
#			norm = minY *0.8
#        		base.SetParameter(0, norm)
		hist1, hist2 = GSC
	        hist1.SetMarkerStyle(8)
#	    	hist1.SetMarkerSize(1.25)
	       	hist1.SetMarkerColor(1)
		if (hist1.Integral() != 0):
	 		hist1.Scale(1/hist1.Integral())
		hist1.SetLineColor(1)
		hist1.SetFillColor(1)
		hist1.SetFillStyle(3007)
		hist2.SetMarkerStyle(24)
#		hist2.SetMarkerSize(1.25)
       		hist2.SetMarkerColor(4)
		if (hist2.Integral() != 0):
			hist2.Scale(1/hist2.Integral())
		hist2.SetFillStyle(3004)
		hist2.SetLineColor(4)
		hist2.SetFillColor(4)

        	histo_maximum = hist1.GetBinContent(hist1.GetMaximumBin()) if hist1.GetBinContent(hist1.GetMaximumBin()) > hist2.GetBinContent(hist2.GetMaximumBin()) else hist2.GetBinContent(hist2.GetMaximumBin())

		if histo_maximum:
			rescale = 0.06/histo_maximum
		else:
			rescale = 0.06
		hist1.Scale(rescale)
		hist2.Scale(rescale)

		hist2.Add(base)
		hist1.Add(base)


	#	hist2.Draw("histsame")
	#	hist1.Draw("histsame")

		l = TLegend(x2, y2-(4.*legend_step),x2+0.44,0.88-(2.*legend_step), "hists")
		l.AddEntry(hist1,labels[0],"fp")
		l.AddEntry(hist2,labels[1],"fp")
		l.SetFillStyle(0)
		l.SetBorderSize(0)
#		l.Draw("same")

	if not ratio:
        	canv.Print(outFile)
		return


	m_hist.GetXaxis().SetLabelColor(0)
	m_hist.GetXaxis().SetTitleColor(0)

	nominal = graphs[0]

	mins = []
	maxes = []
	
	m_ratio = TMultiGraph() if isGraph else THStack("ratio", "ratio")

	maxg = graphs[getGraphMaxX(graphs)] if isGraph else ""
	if isGraph and maxg.GetN():
		maxg_clone = maxg.Clone()
		lastX = maxg.GetX()[maxg_clone.GetN()-1]
		lastEX = maxg.GetEX()[maxg_clone.GetN()-1]
		maxg_clone.Set(0)
		maxg_clone.SetPoint(0, lastX, max*0.99) 
		maxg_clone.SetPointError(0, lastEX, 0) 
		maxg_clone.SetLineColor(0)	
		maxg_clone.SetMarkerColor(0)
		m_ratio.Add(maxg_clone.Clone())	

	#gPad.Update()
	i=1
	h_ratio=""
	skipLimit = not quad and not diff and not noRatioLimit
	all_arrows=[]
	isAllEmpty=True	
	while i < len(graphs):
		if (alternate and i%2) or (alternate2 and (i+1)%2) or (i in skipped):
			i+=1
			continue
		ratio,min,max,arrows,isEmpty = createRatioGraph(nominal, graphs[i], bins=bins, asym=asym, diff=diff, quad=quad, fracErrors=fracErrors, skipLimit=skipLimit, correlated=correlated) if isGraph else createRatioHist(nominal, graphs[i], asym=asym, diff=diff, quad=quad, fracErrors=fracErrors, skipLimit=skipLimit, correlated=correlated) 
		mins.append(min)
		maxes.append(max)
		stylei = i if len(graphs) > 2 else 0
		ratio.SetLineColor(graphs[i].GetLineColor())
		ratio.SetMarkerColor(graphs[i].GetMarkerColor())
		ratio.SetMarkerStyle(graphs[i].GetMarkerStyle())
		m_ratio.Add(ratio.Clone(ratio.GetName() + "_clone"))
		all_arrows.append(arrows)
		if not h_ratio and i:
			h_ratio = ratio
		i+=1
		isAllEmpty = isEmpty and isAllEmpty

	if len(skipped) == (len(graphs)-1) or isAllEmpty:
		h_ratio = graphs[0].Clone(graphs[0].GetName() + "_Clone_ratio")
#		if isGraph: 
#			h_ratio.Set(0)
#		else: 
#			h_ratio.Reset()
#		h_ratio.SetLineColor(0)
#		h_ratio.SetMarkerColor(0)
#		h_ratio.SetMarkerStyle(0)
		m_ratio.Add(h_ratio.Clone(graphs[0].GetName() + "_Clone_ratio_2"))
		mins.append(min)
		maxes.append(max)

#		return h_ratio
	mins.sort()
	maxes.sort()
	min_ratio = mins[0] if mins else -1
	max_ratio = maxes[-1] if maxes else -1
	pad1.cd()

#	print "scaling: ", padScaling	
#        gPad.SetGridy(1)

	pad1.SetFillColor(0)	
	pad1.SetFillStyle(0)	
	m_ratio.Draw(draw_opt)
	ratio_hist = m_ratio if isGraph else m_ratio.GetHistogram()

#	m_hist.GetYaxis().SetTitleSize(m_hist.GetYaxis().GetTitleSize()*1.3)
#	m_hist.GetYaxis().SetLabelSize(m_hist.GetYaxis().GetLabelSize()*1.3)
#	m_hist.GetYaxis().SetTitleOffset(m_hist.GetYaxis().GetTitleOffset()*0.8)



	axis_title_size = style.GetTitleYSize() * BottomScaling
	axis_label_size = style.GetLabelSize("Y") * BottomScaling 
	axis_offset = m_hist.GetYaxis().GetTitleOffset() * ((1. - padScaling)/padScaling) 
	if pt_resol:
		#axis_title_size *= 0.9
		#axis_label_size *= 0.9
		axis_offset *= 1.35
		m_hist.GetYaxis().SetTitleOffset(m_hist.GetYaxis().GetTitleOffset()*1.35)
		
#	axis_title_size *= 1.3
#	axis_label_size *= 1.3
	if pt_resol: pad1.SetLeftMargin(pad1.GetLeftMargin()*1.25)
        ratio_hist.GetXaxis().SetTitle(m.GetXaxis().GetTitle())
        ratio_hist.GetYaxis().SetTitleSize(axis_title_size)
	ratio_hist.GetXaxis().SetTitleSize(axis_title_size)
       	ratio_hist.GetYaxis().SetLabelSize(axis_label_size)
        ratio_hist.GetXaxis().SetLabelSize(axis_label_size)
        ratio_hist.GetYaxis().SetNdivisions(5)
      	ratio_hist.GetYaxis().SetTitleOffset(axis_offset)
        #ratio_hist.GetXaxis().SetTitleOffset(ratio_hist.GetXaxis().GetTitleOffset()*0.8)

	if autoXRange or rangeX:
#		#minX = getMinimumXMulti(graphs, isGraph) 
#		minX = m_hist.GetXaxis().GetXmin() if not isGraph else getMinimumXMulti(graphs, isGraph) 
#		maxX = getMaximumXMulti(graphs, isGraph)
#		if logX: maxX += (maxX-minX)*0.05

		if isGraph: ratio_hist.GetXaxis().SetLimits(minX, maxX)
		else:  ratio_hist.GetXaxis().SetRangeUser(minX, maxX)
		"" if not isGraph else m_ratio.Draw(draw_opt)
		gPad.Update()


	for label in labels:
		if isinstance(label, dict):
			labels[labels.index(label)] = "MC" 

	if "YTHIA" in labels[0] :
		labels[0] = "MC"
		#labels[0] = "P#scale[0.85]{YTHIA}8 Dijet"
	elif "YTHIA" in labels[1]:
		#labels[1] = "P#scale[0.85]{YTHIA}8 Dijet"
		labels[1] = "MC"
		
	
	if quad:
	       	#ratio_hist.GetYaxis().SetTitle("#sqrt{#sigma'^{2} - #sigma^{2}}")
	       	ratio_hist.GetYaxis().SetTitle("sgn(#sigma'-#sigma)#sqrt{#sigma'^{2}-#sigma^{2}}")
		line = TLine(ratio_hist.GetXaxis().GetXmin(), 0, ratio_hist.GetXaxis().GetXmax(), 0)
		#line = TLine(ratio_hist.GetXaxis().GetXmin() if not autoXRange else minX, 0, ratio_hist.GetXaxis().GetXmax() if not autoXRange else maxX, 0)
		draw_line = ratio_hist.GetYaxis().GetXmin() <= 0
	elif diff:
	       	ratio_hist.GetYaxis().SetTitle("Difference")
		line = TLine(ratio_hist.GetXaxis().GetXmin(), 0, ratio_hist.GetXaxis().GetXmax(), 0)
		#line = TLine(ratio_hist.GetXaxis().GetXmin() if not autoXRange else minX, 0, ratio_hist.GetXaxis().GetXmax() if not autoXRange else maxX, 0)
		draw_line = ratio_hist.GetYaxis().GetXmin() <= 0
	elif asym:
		if len(graphs) < 3:
		       	ratio_hist.GetYaxis().SetTitle("(" + labels[1] + " + 2)/(" + labels[0] + " + 2)")
		else:
			ratio_hist.GetYaxis().SetTitle("Ratio + 2")
		line = TLine(ratio_hist.GetXaxis().GetXmin(), 2, ratio_hist.GetXaxis().GetXmax(), 2)
		#line = TLine(ratio_hist.GetXaxis().GetXmin() if not autoXRange else minX, 2, ratio_hist.GetXaxis().GetXmax() if not autoXRange else maxX, 2)
		draw_line = ratio_hist.GetYaxis().GetXmin() <= 2
	else:
		if len(graphs) < 3:
		       	ratio_hist.GetYaxis().SetTitle(labels[1] + "/" + labels[0])
		else:
			if "DATA" in labels[0]: ratio_hist.GetYaxis().SetTitle("MC/Data")
			elif "DATA" in labels[1]:ratio_hist.GetYaxis().SetTitle("MC/Data")
			else:  ratio_hist.GetYaxis().SetTitle("/" + labels[0])

			#ratio_hist.GetYaxis().SetTitle("Ratio to %s"%labels[0])
	
		line = TLine(ratio_hist.GetXaxis().GetXmin(), 1, ratio_hist.GetXaxis().GetXmax(), 1)
		#line = TLine(ratio_hist.GetXaxis().GetXmin() if not autoXRange else minX, 1, ratio_hist.GetXaxis().GetXmax() if not autoXRange else maxX, 1)
		draw_line = ratio_hist.GetYaxis().GetXmin() <= 1

	line.SetLineStyle(2)
	if draw_line: line.Draw("same")
	
	m_ratio_max = m_ratio.GetHistogram() if isGraph else m_ratio


	

	m_ratio_max.SetMinimum(min_ratio - ((max_ratio - min_ratio) * 0.1))
	m_ratio_max.SetMaximum(max_ratio + ((max_ratio - min_ratio)*0.1))

	'''
	if not min_ratio and not quad and not diff and not noRatioLimit:
		m_ratio_max.SetMinimum(0.95)
	else:
		if min_ratio < 0:
			m_ratio_max.SetMinimum(min_ratio*1.05)
		else:
			m_ratio_max.SetMinimum(min_ratio*0.95)

	if max_ratio < 5 or (noRatioLimit and max_ratio < 10) or diff or quad:
	        m_ratio_max.SetMaximum(max_ratio *1.04)
	        #m_ratio_max.SetMaximum(max_ratio + ((10**(math.floor(0 if not max_ratio else math.log10(abs(max_ratio)))-1)) * 2))
	else:
		m_ratio_max.SetMaximum(10.9)	

	'''
	if ratioYRange:
		m_ratio_max.SetMinimum(ratioYRange[0])
		#m_ratio_max.SetMaximum(ratioYRange[1] + ((10**(math.floor(0 if not ratioYRange[1] else math.log10(abs(ratioYRange[1])))-1)) * 2))
		m_ratio_max.SetMaximum(ratioYRange[1] *1.04)
#        h_ratio.SetMinimum(min*0.95)
#        h_ratio.SetMaximum(h_ratio.GetMaximum()*1.05)

	'''
	arrows=[]

	for i in xrange(len(all_arrows)):
		arrows.append(drawArrows(all_arrows[i], graphs[i].GetMarkerColor(), m_ratio_max.GetMaximum() - ((m_ratio_max.GetMaximum() - m_ratio_max.GetMinimum()) * 0.5), m_ratio_max.GetMaximum()))
	'''
	gPad.SetLogx(logX)	
	ratio_hist.GetXaxis().SetMoreLogLabels(logX)
        gPad.Update()
	'''
	if diff and "resp" in outFile and graph1.GetN() > 2  and graph2.GetN() > 2:
		canv.SetName("test")
		
#       	 	myText(0.8, 0.6, 1, "N_{\\rm Segments}")
		canv.Print("2.pdf")	
		canv.Print("t.root")
		exit()
	'''
        #canv.cd(0)
        canv.Print(outFile)
        #TCanvas.Print(outFile)
	if returnRatio:
		return h_ratio 
	del canv

def drawArrows(arrows, colour, y1, y2):
	ars = []
	for arrow in arrows:
		a = TArrow(arrow, y1, arrow, y2)
		print a
		a.SetLineColor(colour)
		a.SetFillColor(colour)
		a.Draw()
		ars.append(a)
	return ars

def compareGraphs(graph1, graph2, outFile, labels,  globalLabels=[], logX=False, logY=False, bins=[], bottom=False, right=False, ratio=True, rangeX=[], quad=False, diff=False, asym=False, returnRatio=False, min=-1, max=-1, ATLAS="", xTitle="", yTitle="", ratioYRange=[], resp=False, GSC=[], noRatioLimit=False, autoXRange=False, extraOpt="", pt_resol=False, correlated=False):
	return comparePlots([graph1, graph2], [1,4],[8,4], outFile, labels,  globalLabels=globalLabels, logX=logX, bins=bins, bottom=bottom, right=right, ratio=ratio, rangeX=rangeX, quad=quad, diff=diff, asym=asym, returnRatio=returnRatio, min=min, max=max, ATLAS=ATLAS, xTitle=xTitle, yTitle=yTitle, ratioYRange=ratioYRange, resp=resp, GSC=GSC, noRatioLimit=noRatioLimit, autoXRange=autoXRange, pt_resol=pt_resol, correlated=correlated)


def compareGraphsGSC(graph1, graph2, hist1, hist2, outFile, labels,  globalLabels=[], logX=False, bins=[], bottom=False, right=False, ratio=True, rangeX=[], quad=False, diff=False, asym=False, returnRatio=False, min=-1, max=-1, ATLAS="", xTitle="", yTitle="", ratioYRange=[], resp=False, noRatioLimit=False, autoXRange=False, correlated=False):
	return compareGraphs(graph1, graph2, outFile, labels,  globalLabels=globalLabels, logX=logX, bins=bins, bottom=bottom, right=right, ratio=ratio, rangeX=rangeX, quad=quad, diff=diff, asym=asym, returnRatio=returnRatio, min=min, max=max, ATLAS=ATLAS, xTitle=xTitle, yTitle=yTitle, ratioYRange=ratioYRange, resp=resp, GSC=[hist1, hist2], noRatioLimit=noRatioLimit, autoXRange=autoXRange, correlated=correlated)
	

def compareHists(hist1, hist2, outFile, labels,  globalLabels=[], logX=False, logY=False, rangeX=[], returnRatio=False, opt="", quad=False, diff=False, GSC=False, ratio=True, ATLAS="", ratioYRange=[], noRatioLimit=False, right=False, resp=False, asym=False, xTitle="", yTitle="", min=-1, max=-1, autoXRange=False, correlated=False):
	return comparePlots([hist1, hist2], [1,4],[8,4], outFile, labels,  globalLabels=globalLabels, logX=logX, logY=logY, right=right, ratio=ratio, rangeX=rangeX, quad=quad, diff=diff, asym=asym, returnRatio=returnRatio, min=min, max=max, ATLAS=ATLAS, xTitle=xTitle, yTitle=yTitle, ratioYRange=ratioYRange, resp=resp, noRatioLimit=noRatioLimit, autoXRange=autoXRange, correlated=correlated)


def compareMultiGraph(hists, outFile, colours, markers, labels, globalLabels=[], logX=False, logY=False, rangeX=[], returnRatio=False, alternate=False, ATLAS="", leftLegend=False, fracErrors=False, minY=-1, ratio=True, asym=False, quad=False, diff=False, alternate2=False, bins=[], ratioYRange=[], resp=False, noRatioLimit=False, min=-1, max=-1, xTitle="", yTitle="", autoXRange=False, correlated=False, LCW=False):
	return comparePlots(hists, colours, markers, outFile, labels,  globalLabels=globalLabels, logX=logX, logY=logY, bins=bins, ratio=ratio, rangeX=rangeX, quad=quad, diff=diff, asym=asym, returnRatio=returnRatio, min=(minY if min==-1 else min), ATLAS=ATLAS, ratioYRange=ratioYRange, resp=resp, noRatioLimit=noRatioLimit, alternate2=alternate2, alternate=alternate, fracErrors=fracErrors, leftLegend=leftLegend, max=max, xTitle=xTitle, yTitle=yTitle, autoXRange=autoXRange, correlated=correlated, LCW=LCW)


def compareMulti(hists, outFile, colours, markers, labels, globalLabels=[], logX=False, logY=False, rangeX=[], returnRatio=False, alternate=False, ATLAS="", leftLegend=False, fracErrors=False, minY=-1, ratio=True, graph=False, alternate2=False, ratioYRange=[], max=-1, resp=False, asym=False, quad=False, diff=False, noRatioLimit=False, xTitle="", yTitle="", autoXRange=False, opt="", correlated=False):
	return comparePlots(hists, colours, markers, outFile, labels,  globalLabels=globalLabels, logX=logX, logY=logY, ratio=ratio, rangeX=rangeX, quad=quad, diff=diff, asym=asym, returnRatio=returnRatio, min=minY, max=max, ATLAS=ATLAS, ratioYRange=ratioYRange, resp=resp, noRatioLimit=noRatioLimit, graph=graph, alternate=alternate, alternate2=alternate2, fracErrors=fracErrors, leftLegend=leftLegend, xTitle=xTitle, yTitle=yTitle, autoXRange=autoXRange, correlated=correlated)
		


def divideHist(hist1, hist2):
	h_ratio = hist1.Clone(hist1.GetName() + "_C")
	h_ratio.Reset() 
#        h_ratio.Divide(hist1)

        ratio = 0
        error = 0
        min = 1
        for ii in range(h_ratio.GetNbinsX()+1):
               if not ii:
                        continue
               if ( hist1.GetBinContent(ii) > 0 ):
                       ratio = hist2.GetBinContent(ii) / hist1.GetBinContent(ii)
                       error = Get_error(ii,hist2,hist1)
               else:
                      ratio = 0
                      error = 0

               h_ratio.SetBinContent(ii,ratio)
               h_ratio.SetBinError(ii,error)
               if ratio and ratio < min:
                       min=ratio
	return h_ratio

def setTitles(hist, x="", y="", z=""):
	if x:
		hist.GetXaxis().SetTitle(x)
	if y:
		hist.GetYaxis().SetTitle(y)
	if z:
		hist.GetZaxis().SetTitle(z)

def makeStackGraph(hists, colours=[], markers=[]):
	y = 0.85	
	stack = []
	for i in xrange(len(hists)):
		hist = hists[i]	
		hist.SetMarkerColor(colours[i])
		hist.SetLineColor(colours[i])
		hist.SetMarkerStyle(markers[i])
		stack.append(hist)
	return stack


def makeStack(hists, colours=[], markers=[]):
	y = 0.85	
	stack = THStack()
	for i in xrange(len(hists)):
		hist = hists[i]	
		hist.SetMarkerColor(colours[i])
		hist.SetLineColor(colours[i])
		hist.SetMarkerStyle(markers[i])
		stack.Add(hist)


def printHist(hist, outName, opt="", binLabels=[], logX=False, logY=False, logZ=False, rangeX=[], rangeY=[], rangeZ=[], min=-1, max=-1, stack=[], x="", y="", z="", binxy=[], asym=False, resp=False):

	set_palette()
	SetAtlasStyle()
	can = TCanvas("c0", "c0", 700, 500)
	can.cd()
	n = 0
	if not isinstance(hist, TObject):
		ranges = getMaxGraphRanges(hist)
		if ranges[0] !=-1:
#			hist.insert(0, hist.pop(ranges[2]))
			while not hist[0].GetN():
				hist.pop(0)
			hist[0].Draw(opt[0])
			hist[0].GetXaxis().SetRangeUser(0, ranges[0]*1.1)
			hist[0].GetYaxis().SetRangeUser(0.6, ranges[1]*1.1)
#			if (not "i" in str(ranges[1])):
#				hist[0].SetMaximum(ranges[1])
#				hist[0].SetMinimum(0.6)
#			else:
#				hist[0].SetMaximum(2)
#				hist[0].SetMinimum(0.6)

#			hist[0].SetMinimum(0.8)
#			hist[0].SetMaximum(1.2)

#			hist[0].SetMinimum(0.4)
		n+=1	
		for hist in hist[1:]:
			if len(stack)==3 and (not n%2):
				hist.Draw(opt[1])
			elif len(stack) != 3:
				hist.Draw(opt[1])
			n+=1
	else:
		hist.Draw(opt)

	if binLabels:
		drawLabels(binLabels, 0.22 if not "colz" in opt and not "text" in opt else 0.15, 0.87, step=0.06)
	
	gPad.SetLogx(logX)
	gPad.SetLogy(logY)
	gPad.SetLogz(logZ)

	if min != -1:
		hist.SetMinimum(min)
	if max!=-1:
		hist.SetMaximum(max)
	if stack:
		y = 0.87
		for i in xrange(len(stack[1])-1):
			if len(stack)==3 and (not i%2):
	        		myMarkerText(0.7, y-(i*0.07)/2, colours[i], markers[i], "%.2f <= %s < %.2f"%(stack[1][i], stack[0], stack[1][i+1]),0.8, 0.02)		
			elif len(stack) !=3:
	        		myMarkerText(0.7, y-i*0.07, colours[i], markers[i], "%.2f <= %s < %.2f"%(stack[1][i], stack[0], stack[1][i+1]),0.8, 0.02)		

	if isinstance(hist, TGraph2D):	
		if not hist.GetN():
			return
		hist.GetXaxis().SetLabelSize(0.03)
		hist.GetYaxis().SetLabelSize(0.03)
		hist.GetZaxis().SetLabelSize(0.03)
		hist.GetXaxis().SetLabelOffset(0.003)
		hist.GetYaxis().SetLabelOffset(0.003)
		hist.GetZaxis().SetLabelOffset(0.003)
		hist.GetXaxis().SetTitleSize(0.05)
		hist.GetYaxis().SetTitleSize(0.05)
		hist.GetZaxis().SetTitleSize(0.05)
	if "colz" in opt:
		hist.GetXaxis().SetTitleOffset(1.2)
		hist.GetYaxis().SetTitleOffset(1.2)
		hist.GetZaxis().SetTitleOffset(1.23)

	setTitles(hist, x, y, z)
        if rangeZ:
		hist.GetZaxis().SetRangeUser(rangeZ[0], rangeZ[1]) 
        if rangeX:
		hist.GetXaxis().SetRangeUser(rangeX[0], rangeX[1]) 

        if rangeY:
		hist.GetYaxis().SetRangeUser(rangeY[0], rangeY[1]) 

	if min != -1:
		hist.SetMinimum(min)
				
	if asym or resp:
		line2 = TLine(hist.GetXaxis().GetXmin(), 0 if asym else 1, hist.GetXaxis().GetXmax(), 0 if asym else 1)
		line2.SetLineStyle(2)
		line2.Draw("same")

	if "colz" in opt: 
	        gPad.SetLeftMargin(0.13)
        	gPad.SetRightMargin(0.2)
	if "text" in opt:
		hist.GetYaxis().SetTitleOffset(1.2)
	        gPad.SetLeftMargin(0.13)
		gStyle.SetPaintTextFormat(".0f")
	#hist.
	gPad.Update()
	can.Print(outName)

def convertToGraph(hist):

	graph = TGraphErrors()
	graph.SetName(hist.GetName() + "_graph")
	
	for bin in xrange(hist.GetNbinsX()):
		graph.SetPoint(bin+1, hist.GetBinCenter(bin+1), hist.GetBinContent(bin+1))
		graph.SetPointError(bin+1, 0, hist.GetBinError(bin+1))

	graph.GetXaxis().SetTitle(hist.GetXaxis().GetTitle())
	graph.GetYaxis().SetTitle(hist.GetYaxis().GetTitle())

	return graph
	

def printBinLabels(binLabels, x1=0.25, y1=0.85):
	for label in binLabels:
		myText(x1,y1, 1, label , 0.03)
		y1 -= 0.07
def createSegPercentagePlots(segPlots, binning, histName):

	hist = TH1D(histName, histName, len(binning)-1, array('d', binning))
#	hist.Sumw2()
	hist.GetXaxis().SetTitle("p_{T}")
	hist.GetYaxis().SetTitle("% Jets with N_{Segments} > 20")
	print "nsbins " + str(len(segPlots))
	for ebin in xrange(len(segPlots)):
		hist.SetBinContent(ebin+1, segPlots[ebin].Integral(6, 15))
		print "ebin : " + str(ebin) + " nx : " + str(segPlots[ebin].GetNbinsX())
		print "ebin : " + str(ebin) + " integral : " + str(segPlots[ebin].Integral(6, 15))

	return hist

def getMaxGraphRanges(graphs):
	maxX = -1
	maxY = -1
	minX = 99999999999999999
	minY = 99999999999999999
	for i in xrange(len(graphs)):
		if graphs[i].GetN() and maxX < TMath.MaxElement(graphs[i].GetN(), graphs[i].GetX()):
			maxX = TMath.MaxElement(graphs[i].GetN(), graphs[i].GetX()) 
			maxXi = i
		if graphs[i].GetN() and maxY < TMath.MaxElement(graphs[i].GetN(), graphs[i].GetY()):
			maxY = TMath.MaxElement(graphs[i].GetN(), graphs[i].GetY()) 
			maxYi = i
		if graphs[i].GetN() and minX > TMath.MinElement(graphs[i].GetN(), graphs[i].GetX()):
			minX = TMath.MinElement(graphs[i].GetN(), graphs[i].GetX()) 
		if graphs[i].GetN() and minY > TMath.MinElement(graphs[i].GetN(), graphs[i].GetY()):
			minY = TMath.MinElement(graphs[i].GetN(), graphs[i].GetY()) 

	return [maxX, maxY, maxXi, maxYi, minX, minY]

