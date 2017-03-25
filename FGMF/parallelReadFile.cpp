#include "parallelReadFile.h"
#include "fgmf.h"   // 用到struct sRateNode

struct edge_parser      // egde相当于rateNode
{
    const char* const strfirst;
    const char* const strlast;
    const char*       crr;

    edge_parser(const char* const first, const char* const last)
        : strfirst(first), strlast(last), crr(first) {}

    template<typename OutputIt>
    void operator()(OutputIt dst)
    {
        while(crr < strlast)
        {
            eat_empty_lines();

            if(crr < strlast)
                *dst++ = eat_edge();
        }
    }

    sRateNode eat_edge()
    {
        const vint s = eat_id();
        eat_separator();
        const vint t = eat_id();
        eat_separator();
        const dweight weight = eat_weight();
        sRateNode edge(s, t, weight);
        setSubBlockIdx(edge);
        setLabel(edge);
        //return edge{ s, t, 1.0 };  // FIXME: edge weight is not supported so far
        return edge;
    }

    vint eat_id()
    {
        //
        // Naive implementation is faster than library functions such as `atoi` and
        // `strtol`
        //
        const auto _crr = crr;
        vint       v = 0;

        for(; crr < strlast && isdigit(*crr); ++crr)
        {
            const vint _v = v * 10 + (*crr - '0');

            if(_v < v)   // overflowed
                std::cerr << "Too large vertex ID at line " << crr_line();

            v = _v;
        }

        if(_crr == crr)   // If any character has not been eaten
            std::cerr << "Invalid vertex ID at line " << crr_line();

        return v;
    }

    dweight eat_weight()
    {
        dweight weight = atof(crr);
        crr = std::find(crr, strlast, '\n');
        return weight;
    }

    void eat_empty_lines()
    {
        while(crr < strlast)
        {
            if(*crr == '\r') ++crr;                                 // Empty line
            else if(*crr == '\n') ++crr;                                 // Empty line
            else if(*crr == '#') crr = std::find(crr, strlast, '\n');   // Comment
            else break;
        }
    }

    void eat_separator()
    {
        while(crr < strlast && (*crr == '\t' || *crr == ',' || *crr == ' '))
            ++crr;
    }

    // Only for error messages
    size_t crr_line()
    {
        return std::count(strfirst, crr, '\n');
    }
};

int parallelReadFile(std::string &graphpath, std::vector<sRateNode> &edges, int ompNumThread)
{
    //graphpath = "E:\\code\\gpu\\test\\sgd-single-kernel\\sgd-single-kernel\\input.txt";
    HANDLE hd = CreateFile(graphpath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    if(hd == INVALID_HANDLE_VALUE)
        cerr << "INVALID_HANDLE_VALUE:" << GetLastError();

    size_t size = GetFileSize(hd, NULL);
    cout << "file size:" << size << "B" << endl;
    HANDLE handle = CreateFileMapping(hd, NULL, PAGE_READONLY, 0, 0, NULL);
    //HANDLE handle = CreateFileMapping(hd, NULL, PAGE_EXECUTE_READ, 0, 100, NULL);
    const char* data = static_cast<char*>(MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0));

    if(data == NULL)
        std::cerr << "map(2): " << GetLastError();

    //const int          nthread = omp_get_max_threads();
	const int          nthread = ompNumThread;
    //const size_t       zchunk = 1024 * 64;  // 64KB
    const size_t       zchunk = 1024 * 1024 * 64;  // 64MiB
    const size_t       nchunk = size / zchunk + (size % zchunk > 0);
    //std::vector<std::deque<sRateNode>> eparts(nthread);
    std::vector<std::vector<sRateNode>> eparts(nthread);
    
	//#pragma omp parallel for schedule(dynamic, 1)
	#pragma omp parallel num_threads(nthread)
	{
	#pragma omp for schedule(dynamic, 1)
    for(int i = 0; i < nchunk; ++i)
    {
        const char* p = data + zchunk * i;
        const char* q = data + min(zchunk * (i + 1), size);

        // Advance pointer `p` to the end of a line because it is possibly at the
        // middle of the line
        if(i > 0) p = std::find(p, q, '\n');

        if(p < q)     // If `p == q`, do nothing
        {
            q = std::find(q, data + size, '\n');  // Advance `q` likewise
            edge_parser(p, q)(std::back_inserter(eparts[omp_get_thread_num()])); //实现在容器尾部插入元素。
        }
    }
	}

    // Compute indices to copy each element of `eparts` to
    std::vector<size_t> eheads(nthread + 1);

    for(int t = 0; t < nthread; ++t)
        eheads[t + 1] = eheads[t] + eparts[t].size();

    // Gather the edges read by each thread to a single array
    //std::vector<sRateNode> edges(eheads.back());
	edges.resize(eheads.back());

	//#pragma omp parallel for schedule(guided, 1)
	#pragma omp parallel num_threads(nthread)
	{
		#pragma omp for schedule(guided, 1)
		for (int t = 0; t < nthread; ++t)
			boost::copy(eparts[t], edges.begin() + eheads[t]);
	}
    cout << "edge size:" << edges.size() << "\n" << endl;

    // Gather the edges read by eac
    if(UnmapViewOfFile(const_cast<char*>(data)) == 0)
        std::cerr << "unmap(2): " << GetLastError();

    CloseHandle(hd);
    CloseHandle(handle);
    printLine();
    //getchar();
    return 0;
    //exit(0);
}