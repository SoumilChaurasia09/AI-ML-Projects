// Stop words to filter out common English terms in tokenization
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 
  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
  'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 
  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
  'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
  'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
  'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
  'they', 'them', 'their', 'this', 'that', 'these', 'those', 'am', 'been', 'being', 'have',
  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'as', 'until', 'while', 'because',
  'what', 'which', 'who', 'whom', 'where', 'why', 'how', 'are', 'is', 'am'
]);

// Helper: Tokenize, clean, and lowercase words
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '') // Remove punctuation except hyphens
    .split(/\s+/)
    .filter(token => token.length > 1 && !STOP_WORDS.has(token));
}

// Chunks raw university document text based on section boundaries and sentence counts
function chunkProspectus(docText, sentenceCount = 2, overlap = 1) {
  const sections = docText.split(/\[Section:\s*/);
  const chunks = [];

  sections.forEach(section => {
    if (!section.trim()) return;

    // Extract section title and body
    const parts = section.split(']');
    const sectionTitle = parts[0].trim();
    const sectionBody = parts.slice(1).join(']').trim();

    if (!sectionBody) return;

    // Split section body into sentences
    const sentences = sectionBody.match(/[^.!?]+[.!?]+(\s|$)/g) || [sectionBody];
    
    // Chunking with sliding window
    for (let i = 0; i < sentences.length; i += (sentenceCount - overlap)) {
      const chunkSentences = sentences.slice(i, i + sentenceCount);
      if (chunkSentences.length === 0) break;

      const chunkText = chunkSentences.join("").trim();
      if (chunkText.length < 15) continue; // Ignore tiny lines

      // RAG Best Practice: prepend chunk text with section title to boost semantic relevance
      const enrichedContent = `[${sectionTitle}] ${chunkText}`;

      chunks.push({
        id: `chunk_${sectionTitle.toLowerCase().replace(/[^a-z0-9]/g, '_')}_${chunks.length}`,
        category: sectionTitle,
        content: enrichedContent,
        rawContent: chunkText
      });

      // Break if we reached the end of the sentences
      if (i + sentenceCount >= sentences.length) break;
    }
  });

  return chunks;
}

// Global active index memory
let activeChunks = [];
let vocabSet = new Set();
let docFrequency = {}; // Term -> Count of documents containing term
let chunkVectors = []; // List of { chunk, tfIdfVector, norm, visualX, visualY }

// Rebuilds vocabulary, IDF index, and chunk vectors
function indexDocuments(chunks) {
  activeChunks = chunks;
  vocabSet = new Set();
  docFrequency = {};
  chunkVectors = [];

  // A. Tokenize all chunks and record Document Frequencies (DF)
  const tokenizedChunks = chunks.map(chunk => {
    const tokens = tokenize(chunk.content);
    const uniqueTokens = new Set(tokens);

    uniqueTokens.forEach(tok => {
      vocabSet.add(tok);
      docFrequency[tok] = (docFrequency[tok] || 0) + 1;
    });

    return { chunk, tokens };
  });

  // B. Calculate Inverse Document Frequency (IDF) for all words
  const N = chunks.length;
  const idf = {};
  vocabSet.forEach(term => {
    // Smoothed IDF
    idf[term] = Math.log(1 + N / (docFrequency[term] || 1)) + 1;
  });

  // C. Calculate TF-IDF vectors
  tokenizedChunks.forEach(({ chunk, tokens }) => {
    const tf = {};
    tokens.forEach(tok => {
      tf[tok] = (tf[tok] || 0) + 1;
    });

    const totalWords = tokens.length || 1;
    const vector = {};

    vocabSet.forEach(term => {
      if (tf[term]) {
        const normalizedTf = tf[term] / totalWords;
        vector[term] = normalizedTf * idf[term];
      } else {
        vector[term] = 0;
      }
    });

    // Calculate L2 Norm (vector length) for cosine normalization
    let sumSquares = 0;
    Object.values(vector).forEach(val => { sumSquares += val * val; });
    const norm = Math.sqrt(sumSquares) || 1;

    // Deterministic 2D coordinates projection based on category anchors
    const anchors = {
      'Admissions & Eligibility': { x: 90, y: 70 },
      'Tuition Fees & Financial Aid': { x: 230, y: 70 },
      'Hostel & Mess Facilities': { x: 370, y: 140 },
      'Academic Branches & Specializations': { x: 90, y: 210 },
      'Placement & Internship Records': { x: 230, y: 210 }
    };

    const anchor = anchors[chunk.category] || { x: 200, y: 140 };

    // Unique spread offset using simple string hash of the chunk ID
    let hash = 0;
    for (let charIdx = 0; charIdx < chunk.id.length; charIdx++) {
      hash = chunk.id.charCodeAt(charIdx) + ((hash << 5) - hash);
    }
    const angle = (Math.abs(hash) % 360) * Math.PI / 180;
    const distance = 15 + (Math.abs(hash) % 25);

    chunkVectors.push({
      chunk,
      vector,
      norm,
      visualX: anchor.x + Math.cos(angle) * distance,
      visualY: anchor.y + Math.sin(angle) * distance
    });
  });
}

// Retrieves top K documents using Cosine Similarity matching
function retrieveRelevantChunks(queryText, topK = 3) {
  const queryTokens = tokenize(queryText);
  if (queryTokens.length === 0 || activeChunks.length === 0) {
    // Fallback if query or chunks are empty
    return activeChunks.slice(0, topK).map(chunk => ({ chunk, score: 0, visualX: 200, visualY: 140 }));
  }

  // A. Build Query Vector
  const queryTf = {};
  queryTokens.forEach(tok => {
    queryTf[tok] = (queryTf[tok] || 0) + 1;
  });

  const totalQueryWords = queryTokens.length;
  const queryVector = {};
  const N = activeChunks.length;

  vocabSet.forEach(term => {
    if (queryTf[term]) {
      const normalizedTf = queryTf[term] / totalQueryWords;
      const termDf = docFrequency[term] || 0;
      const termIdf = Math.log(1 + N / (termDf || 1)) + 1;
      queryVector[term] = normalizedTf * termIdf;
    } else {
      queryVector[term] = 0;
    }
  });

  // Query vector length (L2 Norm)
  let querySumSquares = 0;
  Object.values(queryVector).forEach(val => { querySumSquares += val * val; });
  const queryNorm = Math.sqrt(querySumSquares) || 1;

  // B. Compute Cosine Similarity between Query Vector and each Document Vector
  const results = chunkVectors.map(item => {
    let dotProduct = 0;
    queryTokens.forEach(term => {
      if (item.vector[term]) {
        dotProduct += queryVector[term] * item.vector[term];
      }
    });

    const similarity = dotProduct / (queryNorm * item.norm);
    return {
      chunk: item.chunk,
      score: isNaN(similarity) ? 0 : similarity,
      visualX: item.visualX,
      visualY: item.visualY
    };
  });

  // C. Sort descending by score and slice top K
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}

// Helper: Get active vocab stats and term weight breakdowns
function getIndexingTelemetry() {
  return {
    totalChunksCount: activeChunks.length,
    vocabularySize: vocabSet.size,
    wordDocumentFrequencies: docFrequency,
    vectors: chunkVectors.map(v => ({
      id: v.chunk.id,
      category: v.chunk.category,
      content: v.chunk.content,
      visualX: v.visualX,
      visualY: v.visualY,
      topTerms: Object.entries(v.vector)
        .filter(([_, val]) => val > 0)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5) // Send top 5 descriptive terms
        .map(([term, weight]) => ({ term, weight }))
    }))
  };
}

// Bind to window for global browser availability
window.tokenize = tokenize;
window.chunkProspectus = chunkProspectus;
window.indexDocuments = indexDocuments;
window.retrieveRelevantChunks = retrieveRelevantChunks;
window.getIndexingTelemetry = getIndexingTelemetry;
