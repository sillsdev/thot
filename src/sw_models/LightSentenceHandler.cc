/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/
 
/********************************************************************/
/*                                                                  */
/* Module: LightSentenceHandler                                     */
/*                                                                  */
/* Definitions file: LightSentenceHandler.cc                        */
/*                                                                  */
/********************************************************************/


//--------------- Include files --------------------------------------

#include "LightSentenceHandler.h"

//--------------- Global variables -----------------------------------


//--------------- Function declarations 


//--------------- Constants


//--------------- Classes --------------------------------------------


//--------------- LightSentenceHandler class function definitions

//-------------------------
LightSentenceHandler::LightSentenceHandler(void)
{
  nsPairsInFiles=0;
  countFileExists=false;
  currFileSentIdx=0;
}

//-------------------------
bool LightSentenceHandler::readSentencePairs(const char *srcFileName,
                                             const char *trgFileName,
                                             const char *sentCountsFile,
                                             pair<unsigned int,unsigned int>& sentRange,
                                             int verbose/*=0*/)
{
      // Clear sentence handler
 if(verbose)
   cerr<<"Initializing sentence handler..."<<endl;
 clear();
  
     // Fill first field of sentRange
 sentRange.first=0;

     // Open source file
 if(awkSrc.open(srcFileName)==ERROR)
 {
   if(verbose)
     cerr<<"Error in source language file: "<<srcFileName<<endl;
   return ERROR;
 }
 else
 {
       // Open target file
   if(awkTrg.open(trgFileName)==ERROR)
   {
     if(verbose)
       cerr<<"Error in target language file: "<<trgFileName<<endl;
     return ERROR;
   }
   else
   {
         // Open file with sentence counts
     if(strlen(sentCountsFile)==0)
     {
           // sentCountsFile is empty
       countFileExists=false;
     }
     else
     {
           // sentCountsFile is not empty
       if(awkSrcTrgC.open(sentCountsFile)==ERROR)
       {
         if(verbose)
           cerr<<"File with sentence counts "<<sentCountsFile<<" does not exist"<<endl;
         countFileExists=false;
       }
       else
         countFileExists=true;
     }
     
         // Read sentence pairs
     if(verbose)
     {
       cerr<<"Reading sentence pairs from files: "<<srcFileName<<" and "<<trgFileName<<endl;
       if(countFileExists) cerr<<"Reading sentence pair counts from file "<<sentCountsFile<<endl;
     }

     while(awkSrc.getln())
     {
       if(!awkTrg.getln())
       {
         if(verbose)
           cerr<<"Error: the number of source and target sentences differ!"<<endl;
         return ERROR;
       }

           // Display warnings if sentences are empty
       if(verbose)
       {
         if(awkSrc.NF==0)
           cerr<<"Warning: source sentence "<<nsPairsInFiles<<" is empty"<<endl;
         if(awkTrg.NF==0)
           cerr<<"Warning: target sentence "<<nsPairsInFiles<<" is empty"<<endl;
       }

       nsPairsInFiles+=1;
     }
         // Print statistics
     if(verbose && nsPairsInFiles>0)
       cerr<<"#Sentence pairs in files: "<<nsPairsInFiles<<endl;
   }
       // Fill second field of sentRange
   sentRange.second=nsPairsInFiles-1;

       // Rewind files
   rewindFiles();
     
   return OK;	
 }
}

//-------------------------
void LightSentenceHandler::rewindFiles(void)
{
      // Rewind files
  awkSrc.rwd();
  awkTrg.rwd();
  awkSrcTrgC.rwd();

      // Read first entry
  getNextLineFromFiles();

      // Reset currFileSentIdx
  currFileSentIdx=0;
}

//-------------------------
void LightSentenceHandler::addSentPair(Vector<std::string> srcSentStr,
                                       Vector<std::string> trgSentStr,
                                       Count c,
                                       const WordAligMatrix& waMatrix,
                                       pair<unsigned int,unsigned int>& sentRange)
{
      // Fill sentRange information
  sentRange.first=nsPairsInFiles+sentPairCont.size();
  sentRange.second=sentRange.first;
      // add to sentPairCont
  sentPairCont.push_back(make_pair(srcSentStr,trgSentStr));
      // add to sentPairCount
  sentPairCount.push_back(c);
  sentPairWaMatrix.push_back(waMatrix);

      // Display warnings if sentences are empty
  if(srcSentStr.empty())
    cerr<<"Warning: source sentence "<<sentRange.first<<" is empty"<<endl;
  if(trgSentStr.empty())
    cerr<<"Warning: target sentence "<<sentRange.first<<" is empty"<<endl;
}

//-------------------------
unsigned int LightSentenceHandler::numSentPairs(void)
{
  return nsPairsInFiles+sentPairCont.size();
}

//-------------------------
int LightSentenceHandler::nthSentPair(unsigned int n,
                                      Vector<std::string>& srcSentStr,
                                      Vector<std::string>& trgSentStr,
                                      Count& c,
                                      WordAligMatrix& waMatrix)
{
  if(n>=numSentPairs())
    return ERROR;
  else
  {
    if(n<nsPairsInFiles)
    {
      return nthSentPairFromFiles(n,srcSentStr,trgSentStr,c);
    }
    else
    {
      size_t vecIdx=n-nsPairsInFiles;
      
      srcSentStr=sentPairCont[vecIdx].first;

      trgSentStr=sentPairCont[vecIdx].second;
      
      c=sentPairCount[vecIdx];

      waMatrix=sentPairWaMatrix[vecIdx];
      
      return OK;
    }
  }
}

//-------------------------
int LightSentenceHandler::nthSentPairFromFiles(unsigned int n,
                                               Vector<std::string>& srcSentStr,
                                               Vector<std::string>& trgSentStr,
                                               Count& c)

{
      // Check if entry is contained in files
  if(n>=nsPairsInFiles)
    return ERROR;
  
      // Find corresponding entries
  if(currFileSentIdx>n)
      rewindFiles();

  if(currFileSentIdx!=n)
  {
    while(getNextLineFromFiles())
    {
      if(currFileSentIdx==n) break;
    }
  }

      // Reset variables
  srcSentStr.clear();
  trgSentStr.clear();

      // Extract information
  for(unsigned int i=1;i<=awkSrc.NF;++i) 
  {
    srcSentStr.push_back(awkSrc.dollar(i));
  }
  for(unsigned int i=1;i<=awkTrg.NF;++i)
  {
    trgSentStr.push_back(awkTrg.dollar(i));
  }

  if(countFileExists)
  {
    c=atof(awkSrcTrgC.dollar(1).c_str());
  }
  else
  {
    c=1;
  }
    
  return OK;
}

//-------------------------
bool LightSentenceHandler::getNextLineFromFiles(void)
{
  bool ret;
  
  ret=awkSrc.getln();
  if(ret==false) return false;

  ret=awkTrg.getln();
  if(ret==false) return false;

  if(countFileExists)
  {
    ret=awkSrcTrgC.getln();
    if(ret==false) return false;
  }

  ++currFileSentIdx;

  return true;
}

//-------------------------
int LightSentenceHandler::getSrcSent(unsigned int n,
                                     Vector<std::string>& srcSentStr)
{
  Vector<std::string> trgSentStr;
  Count c;
  WordAligMatrix waMatrix;

  int ret=nthSentPair(n,srcSentStr,trgSentStr,c,waMatrix);

  return ret;  
}

//-------------------------
int LightSentenceHandler::getTrgSent(unsigned int n,
                                     Vector<std::string>& trgSentStr)
{
  Vector<std::string> srcSentStr;
  Count c;
  WordAligMatrix waMatrix;

  int ret=nthSentPair(n,srcSentStr,trgSentStr,c,waMatrix);

  return ret;  
}

//-------------------------
int LightSentenceHandler::getCount(unsigned int n,
                                   Count& c)
{
  Vector<std::string> srcSentStr;
  Vector<std::string> trgSentStr;
  WordAligMatrix waMatrix;

  int ret=nthSentPair(n,srcSentStr,trgSentStr,c,waMatrix);

  return ret;  
}
//-------------------------
int LightSentenceHandler::getWaMatrix(unsigned int n,
                                      WordAligMatrix& waMatrix)
{
  Vector<std::string> srcSentStr;
  Vector<std::string> trgSentStr;
  Count c;

  int ret=nthSentPair(n,srcSentStr,trgSentStr,c,waMatrix);

  return ret;  
}

//-------------------------
bool LightSentenceHandler::printSentPairs(const char *srcSentFile,
                                          const char *trgSentFile,
                                          const char *sentCountsFile)
{
  ofstream srcOutF;
  ofstream trgOutF;
  ofstream countsOutF;

      // Open file with source sentences
  srcOutF.open(srcSentFile,ios::out);
  if(!srcOutF)
  {
    cerr<<"Error while printing file with source sentences."<<endl;
    return ERROR;
  }

      // Open file with target sentences
  trgOutF.open(trgSentFile,ios::out);
  if(!trgOutF)
  {
    cerr<<"Error while printing file with target sentences."<<endl;
    return ERROR;
  }

      // Open file with sentence counts
  countsOutF.open(sentCountsFile,ios::out);
  if(!countsOutF)
  {
    cerr<<"Error while printing file with sentence counts."<<endl;
    return ERROR;
  }

  for(unsigned int n=0;n<numSentPairs();++n)
  {
    Vector<std::string> srcSentStr;
    Vector<std::string> trgSentStr;
    Count c;
    WordAligMatrix waMatrix;

    nthSentPair(n,srcSentStr,trgSentStr,c,waMatrix);
    
        // print source sentence
    for(unsigned int j=0;j<srcSentStr.size();++j)
    {
      srcOutF<<srcSentStr[j];
      if(j<srcSentStr.size()-1) srcOutF<<" ";
    }
    srcOutF<<endl;
      
        // print target sentence
    for(unsigned int j=0;j<trgSentStr.size();++j)
    {
      trgOutF<<trgSentStr[j];
      if(j<trgSentStr.size()-1) trgOutF<<" ";
    }
    trgOutF<<endl;
      
        // print count
    countsOutF<<c<<endl;
  }

    
      // Close output streams
  srcOutF.close();
  trgOutF.close();
  countsOutF.close();
  
  return OK;
}

//-------------------------
void LightSentenceHandler::clear(void)
{
  sentPairCont.clear();
  sentPairCount.clear();
  sentPairWaMatrix.clear();
  nsPairsInFiles=0;
  awkSrc.close();
  awkTrg.close();
  awkSrcTrgC.close();
  countFileExists=false;
  currFileSentIdx=0;
}
