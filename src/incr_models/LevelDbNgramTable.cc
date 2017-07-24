/*
thot package for statistical machine translation
 
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
/* Module: LevelDbNgramTable                                        */
/*                                                                  */
/* Definitions file: LevelDbNgramTable.cc                           */
/*                                                                  */
/********************************************************************/


//--------------- Include files --------------------------------------

#include "LevelDbNgramTable.h"

//--------------- Function definitions

//-------------------------
LevelDbNgramTable::LevelDbNgramTable(void)
{
    options.create_if_missing = true;
    options.max_open_files = 4000;
    options.filter_policy = leveldb::NewBloomFilterPolicy(48);
    options.block_cache = leveldb::NewLRUCache(100 * 1048576);  // 100 MB for cache
    db = NULL;
    dbName = "";
}

//-------------------------
string LevelDbNgramTable::vectorToString(const Vector<WordIndex>& vec)const
{
    Vector<WordIndex> str;
    for(size_t i = 0; i < vec.size(); i++) {
        // Use WORD_INDEX_MODULO_BYTES bytes to encode index
        for(int j = WORD_INDEX_MODULO_BYTES - 1; j >= 0; j--) {
            str.push_back(1 + (vec[i] / (unsigned int) pow(WORD_INDEX_MODULO_BASE, j) % WORD_INDEX_MODULO_BASE));
        }
    }

    string s(str.begin(), str.end());

    return s;
}

//-------------------------
Vector<WordIndex> LevelDbNgramTable::stringToVector(const string s)const
{
    Vector<WordIndex> vec;

    for(size_t i = 0; i < s.size();)  // A string length is WORD_INDEX_MODULO_BYTES * n + 1
    {
        unsigned int wi = 0;
        for(int j = WORD_INDEX_MODULO_BYTES - 1; j >= 0; j--, i++) {
            wi += (((unsigned char) s[i]) - 1) * (unsigned int) pow(WORD_INDEX_MODULO_BASE, j);
        }

        vec.push_back(wi);
    }

    return vec;
}

//-------------------------
string LevelDbNgramTable::vectorToKey(const Vector<WordIndex>& vec)const
{
    return vectorToString(vec);
}

//-------------------------
Vector<WordIndex> LevelDbNgramTable::keyToVector(const string key)const
{
    return stringToVector(key);
}

//-------------------------
bool LevelDbNgramTable::retrieveData(const Vector<WordIndex>& phrase, int &count)const
{
    if (phrase.size() == 0)
    {
        count = srcInfoNull.get_c_s();

        return true;
    }
    else
    {
        string value_str;
        count = 0;
        string key = vectorToString(phrase);
        
        leveldb::Status result = db->Get(leveldb::ReadOptions(), key, &value_str);  // Read stored src value

        if (result.ok()) {
            count = atoi(value_str.c_str());
            return true;
        } else {
            return false;
        }
    }
}

//-------------------------
bool LevelDbNgramTable::storeData(const Vector<WordIndex>& phrase, int count)
{
    if (phrase.size() == 0)
    {
        srcInfoNull = count;

        return true;
    }
    else
    {
        stringstream ss;
        ss << count;
        string count_str = ss.str();

        leveldb::WriteBatch batch;
        batch.Put(vectorToString(phrase), count_str);
        leveldb::Status s = db->Write(leveldb::WriteOptions(), &batch);

        if(!s.ok())
            cerr << "Storing data status: " << s.ToString() << endl;

        return s.ok();
    }
}

//-------------------------
bool LevelDbNgramTable::init(string levelDbPath)
{
    cerr << "Initializing LevelDB phrase table" << endl;

    if(db != NULL)
    {
        delete db;
        db = NULL;
    }

    if(load(levelDbPath) != OK)
        return ERROR;

    clear();

    return OK;
}

//-------------------------
bool LevelDbNgramTable::drop()
{
    if(db != NULL)
    {
        delete db;
        db = NULL;
    }

    leveldb::Status status = leveldb::DestroyDB(dbName, options);

    if(status.ok())
    {
        return OK;
    }
    else
    {
        cerr << "Dropping database status: " << status.ToString() << endl;
        
        return ERROR;
    }
}

//-------------------------
bool LevelDbNgramTable::load(string levelDbPath)
{
    if(db != NULL)
    {
        delete db;
        db = NULL;
    }

    dbName = levelDbPath;
    leveldb::Status status = leveldb::DB::Open(options, dbName, &db);

    return (status.ok()) ? OK : ERROR;
}

//-------------------------
Vector<WordIndex> LevelDbNgramTable::getSrcTrg(const Vector<WordIndex>& s,
                                               const WordIndex& t)const
{
    // Concatenate vectors s and t
    Vector<WordIndex> st = s;
    st.push_back(t);

    return st;
}


//-------------------------
bool LevelDbNgramTable::getNbestForSrc(const Vector<WordIndex>& s,
                                       NbestTableNode<WordIndex>& nbt)
{
      // TO-BE-DONE (LOW PRIORITY)
}
//-------------------------
bool LevelDbNgramTable::getNbestForTrg(const WordIndex& t,
                                       NbestTableNode<Vector<WordIndex> >& nbt,
                                       int N)
{
    /*LevelDbNgramTable::SrcTableNode::iterator iter;	

    bool found;
    Count t_count;
    LevelDbNgramTable::SrcTableNode node;
    LgProb lgProb;

    nbt.clear();

    found = getEntriesForTarget(t, node);
    t_count = cTrg(t);

    if(found) {
        // Generate transTableNode
        for(iter = node.begin(); iter != node.end(); iter++) 
        {
            Vector<WordIndex> s = iter->first;
            PhrasePairInfo ppi = (PhrasePairInfo) iter->second;
            lgProb = log((float) ppi.second.get_c_st() / (float) t_count);
            nbt.insert(lgProb, s); // Insert pair <log probability, source phrase>
        }

#   ifdef DO_STABLE_SORT_ON_NBEST_TABLE
        // Performs stable sort on n-best table, this is done to ensure
        // that the n-best lists generated by cache models and
        // conventional models are identical. However this process is
        // time consuming and must be avoided if possible
        nbt.stableSort();
#   endif

        while(nbt.size() > (unsigned int) N && N >= 0)
        {
            // node contains N inverse translations, remove last element
            nbt.removeLastElement();
        }

        return true;
    }
    else
    {
        // Cannot find the target phrase
        return false;
    }*/
}

//-------------------------
void LevelDbNgramTable::addTableEntry(const Vector<WordIndex>& s,
                                      const WordIndex& t,
                                      im_pair<Count,Count> inf) 
{
    addSrcInfo(s, inf.first);  // (USUSED_WORD, s)
    addSrcTrgInfo(s, t, inf.second);  // (t, UNUSED_WORD, s)
}

//-------------------------
void LevelDbNgramTable::addSrcInfo(const Vector<WordIndex>& s,
                                   Count s_inf)
{
    storeData(s, (int) s_inf.get_c_s());
}

//-------------------------
void LevelDbNgramTable::addSrcTrgInfo(const Vector<WordIndex>& s,
                                      const WordIndex& t,
                                      Count st_inf)
{
    storeData(getSrcTrg(s, t), (int) st_inf.get_c_st());  // (s, t)
}

//-------------------------
void LevelDbNgramTable::incrCountsOfEntryLog(const Vector<WordIndex>& s,
                                             const WordIndex& t,
                                             LogCount lc) 
{
    // Retrieve previous states
    Count s_count = cSrc(s);
    Count src_trg_count = cSrcTrg(s, t);

    // Update counts
    addSrcInfo(s, s_count + lc.get_c_s());  // (USUSED_WORD, s)
    addSrcTrgInfo(s, t, (src_trg_count + lc.get_c_st()).get_c_st());
}

//-------------------------
im_pair<Count, Count> LevelDbNgramTable::infSrcTrg(const Vector<WordIndex>& s,
                                            const WordIndex& t,
                                            bool& found) 
{
    im_pair<Count, Count> ppi;

    ppi.first = getSrcInfo(s, found);
    if (!found)
    {
        ppi.second = 0;
        return ppi;
    }
    else
    {
        ppi.second = getSrcTrgInfo(s, t, found);
        return ppi;
    }
}

//-------------------------
Count LevelDbNgramTable::getInfo(const Vector<WordIndex>& key,
                                 bool &found)
{
    int count;
    found = retrieveData(key, count);

    Count result = (found) ? Count((float) count) : Count();

    return result;
}

//-------------------------
Count LevelDbNgramTable::getSrcInfo(const Vector<WordIndex>& s,
                                    bool &found)
{
    return getInfo(s, found);
}

//-------------------------
Count LevelDbNgramTable::getTrgInfo(const WordIndex& t,
                                    bool &found)
{
    // Retrieve counter state
    Vector<WordIndex> t_vec;
    t_vec.push_back(t);

    return getInfo(t_vec, found);
}

//-------------------------
Count LevelDbNgramTable::getSrcTrgInfo(const Vector<WordIndex>& s,
                                       const WordIndex& t,
                                       bool &found)
{
    // Retrieve counter state
    return getInfo(getSrcTrg(s, t), found);
}

//-------------------------
Prob LevelDbNgramTable::pTrgGivenSrc(const Vector<WordIndex>& s,
                                     const WordIndex& t)
{
    // Calculates p(t|s)=count(s,t)/count(s)
    Count st_count = cSrcTrg(s, t);
    if ((float) st_count > 0)
    {
        bool found;
        Count s_count = getSrcInfo(s, found);
        if ((float) s_count > 0)
            return (float) st_count.get_c_st() / (float) s_count.get_c_s();
        else
            return 0;
    }
    else return 0;
}

//-------------------------
LgProb LevelDbNgramTable::logpTrgGivenSrc(const Vector<WordIndex>& s,
                                          const WordIndex& t)
{
    Prob p = pTrgGivenSrc(s, t);

    if ((double) p == 0.0)
        return SMALL_LG_NUM;
    
    return p.get_lp();
}

//-------------------------
Prob LevelDbNgramTable::pSrcGivenTrg(const Vector<WordIndex>& s,
                                     const WordIndex& t)
{
    // p(s|t)=count(s,t)/count(t)
    Count st_count = cSrcTrg(s, t);
    if ((float) st_count > 0)
    {
        Count t_count = cTrg(t);
        if ((float) t_count > 0)
            return (float) st_count.get_c_st() / (float) t_count.get_c_s();
        else
            return SMALL_LG_NUM;
    }
    else return SMALL_LG_NUM;
}

//-------------------------
LgProb LevelDbNgramTable::logpSrcGivenTrg(const Vector<WordIndex>& s,
                                          const WordIndex& t)
{
    return pSrcGivenTrg(s,t).get_lp();
}

//-------------------------
bool LevelDbNgramTable::getEntriesForTarget(const WordIndex& t,
                                            LevelDbNgramTable::SrcTableNode& srctn)
{
    /*bool found;

    Vector<WordIndex> start_vec = t;
    start_vec.push_back(UNUSED_WORD);

    Vector<WordIndex> end_vec(t);
    end_vec.push_back(3);

    string start_str = vectorToKey(start_vec);
    string end_str = vectorToKey(end_vec);

    leveldb::Slice start = start_str;
    leveldb::Slice end = end_str;

    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    
    srctn.clear();  // Make sure that structure does not keep old values
    
    int i = 0;
    for(it->Seek(start); it->Valid() && it->key().ToString() < end.ToString(); it->Next(), i++) {
        Vector<WordIndex> vec = keyToVector(it->key().ToString());
        Vector<WordIndex> src(vec.begin() + start_vec.size(), vec.end());

        PhrasePairInfo ppi = infSrcTrg(src, t, found);
        if (!found || (int) ppi.first.get_c_s() == 0 || (int) ppi.second.get_c_s() == 0)
            continue;

        srctn.insert(pair<Vector<WordIndex>, PhrasePairInfo>(src, ppi));
    }

    found = it->status().ok();

    delete it;

    return i > 0 && found;*/
}

//-------------------------
bool LevelDbNgramTable::getEntriesForSource(const Vector<WordIndex>& s,
                                            LevelDbNgramTable::TrgTableNode& trgtn) 
{
    bool found;
    pair<WordIndex,im_pair<Count,Count> > pdp;

    Count s_count = cSrc(s);

    Vector<WordIndex> end_vec = s;
    end_vec[end_vec.size() - 1] += 1;

    string start_str = vectorToKey(s);
    string end_str = vectorToKey(end_vec);

    leveldb::Slice start = start_str;
    leveldb::Slice end = end_str;

    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    
    trgtn.clear();  // Make sure that structure does not keep old values
    
    int i = 0;
    for(it->Seek(start); it->Valid() && it->key().ToString() < end.ToString(); it->Next(), i++) {
        Vector<WordIndex> vec = keyToVector(it->key().ToString());

        if ( s.size() == vec.size() - 1)
        {
            pdp.first = vec.back();  // t
            pdp.second.first = s_count;  // COUNT(s)
            pdp.second.second = Count(atoi(it->value().ToString().c_str()));  // COUNT(t|s)


            if ((int) pdp.second.first.get_c_s() == 0 || (int) pdp.second.second.get_c_st() == 0)
                continue;

            trgtn.insert(pdp);
        }
        
    }

    found = it->status().ok();

    delete it;

    return i > 0 && found;
}

//-------------------------
Count LevelDbNgramTable::cSrcTrg(const Vector<WordIndex>& s,
                                 const WordIndex& t)
{
    bool found;
    return getSrcTrgInfo(s, t, found);
}

//-------------------------
Count LevelDbNgramTable::cSrc(const Vector<WordIndex>& s)
{
    bool found;
    return getSrcInfo(s, found);
}

//-------------------------
Count LevelDbNgramTable::cTrg(const WordIndex& t)
{
    // TODO - Not mandatory
    //bool found;
    //return getTrgInfo(t, found);
}

//-------------------------
LogCount LevelDbNgramTable::lcSrcTrg(const Vector<WordIndex>& s,const WordIndex& t)
{
    // TODO
}

//-------------------------
LogCount LevelDbNgramTable::lcSrc(const Vector<WordIndex>& s)
{
    // TODO
}

//-------------------------
LogCount LevelDbNgramTable::lcTrg(const WordIndex& t)
{
    // TODO
}

//-------------------------
size_t LevelDbNgramTable::size(void)
{
    size_t len = 0;  // TODO: or maybe srcInfoNull.get_c_s()

    for(LevelDbNgramTable::const_iterator iter = begin(); iter != end(); iter++, len++)
    {
        // Do nothing; iterates only over the elements in trie
    }

    return len;
}

//-------------------------
void LevelDbNgramTable::print(bool printString)
{
    cout << "levelDB content:" << endl;   
    for(LevelDbNgramTable::const_iterator iter = begin(); iter != end(); iter++)
    {
        pair<Vector<WordIndex>, Count> x = *iter;
        if (printString) {
            for(size_t i = 0; i < x.first.size(); i++)
                cout << x.first[i] << " ";
        } else {
            cout << vectorToKey(x.first);
        }
        
        cout << ":\t" << x.second.get_c_s() << endl;
    }
}

//-------------------------
void LevelDbNgramTable::clear(void)
{
    if(dbName.size() > 0)
    {
        bool dropStatus = drop();

        if(dropStatus == ERROR)
        {
            exit(2);
        }

        leveldb::Status status = leveldb::DB::Open(options, dbName, &db);
        
        if(!status.ok())
        {
            cerr << "Cannot create new levelDB in " << dbName << endl;
            cerr << "Returned status: " << status.ToString() << endl;
            exit(3);
        }
    }

    srcInfoNull = Count();
}

//-------------------------
LevelDbNgramTable::~LevelDbNgramTable(void)
{
    if(db != NULL)
        delete db;

    if(options.filter_policy != NULL)
        delete options.filter_policy;

    if(options.block_cache != NULL)
        delete options.block_cache;
}

//-------------------------
LevelDbNgramTable::const_iterator LevelDbNgramTable::begin(void)const
{
    leveldb::Iterator *local_iter = db->NewIterator(leveldb::ReadOptions());
    local_iter->SeekToFirst();

    if(!local_iter->Valid()) {
        delete local_iter;
        local_iter = NULL;
    }

    LevelDbNgramTable::const_iterator iter(this, local_iter);

    return iter;
}

//-------------------------
LevelDbNgramTable::const_iterator LevelDbNgramTable::end(void)const
{
    LevelDbNgramTable::const_iterator iter(this, NULL);

    return iter;
}

// const_iterator function definitions
//--------------------------
bool LevelDbNgramTable::const_iterator::operator++(void) //prefix
{
    internalIter->Next();

    bool isValid = internalIter->Valid();

    if(!isValid)
    {
        delete internalIter;
        internalIter = NULL;
    }

    return isValid;
}

//--------------------------
bool LevelDbNgramTable::const_iterator::operator++(int)  //postfix
{
    return operator++();
}

//--------------------------
int LevelDbNgramTable::const_iterator::operator==(const const_iterator& right)
{
    return (ptPtr == right.ptPtr && internalIter == right.internalIter);
}

//--------------------------
int LevelDbNgramTable::const_iterator::operator!=(const const_iterator& right)
{
    return !((*this) == right);
}

//--------------------------
pair<Vector<WordIndex>, Count> LevelDbNgramTable::const_iterator::operator*(void)
{
    return *operator->();
}

//--------------------------
const pair<Vector<WordIndex>, Count>*
LevelDbNgramTable::const_iterator::operator->(void)
{
    string key = internalIter->key().ToString();
    Vector<WordIndex> key_vec = ptPtr->keyToVector(key);

    int count = atoi(internalIter->value().ToString().c_str());

    dataItem = make_pair(key_vec, Count(count));

    return &dataItem;
}

//-------------------------