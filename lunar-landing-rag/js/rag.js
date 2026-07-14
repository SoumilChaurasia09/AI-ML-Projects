// Import omitted for browser script compatibility

// Predefined stop words to filter out noisy terms
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 
  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
  'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 
  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
  'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
  'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
  'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
  'they', 'them', 'their', 'this', 'that', 'these', 'those', 'am', 'been', 'being', 'have',
  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'as', 'until', 'while', 'because'
]);

// Helper: Tokenize and clean text
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '') // Remove punctuation except hyphens
    .split(/\s+/)
    .filter(token => token.length > 1 && !STOP_WORDS.has(token));
}

// Chunks a single document
function chunkDocument(doc, sentenceCount = 2, overlap = 1) {
  // Split by sentences using regex
  const sentences = doc.content.match(/[^.!?]+[.!?]+(\s|$)/g) || [doc.content];
  const chunks = [];
  
  for (let i = 0; i < sentences.length; i += (sentenceCount - overlap)) {
    const chunkSentences = sentences.slice(i, i + sentenceCount);
    if (chunkSentences.length === 0) break;
    
    const chunkText = chunkSentences.join("").trim();
    if (chunkText.length < 15) continue; // Skip tiny fragments
    
    chunks.push({
      id: `${doc.id}_chunk_${chunks.length}`,
      docId: doc.id,
      docTitle: doc.title,
      category: doc.category,
      content: chunkText
    });
    
    // Stop if we have reached the end
    if (i + sentenceCount >= sentences.length) break;
  }
  
  return chunks;
}

// Global active corpus and indexing state
let allChunks = [];
let vocabulary = new Set();
let df = {}; // Document Frequency for IDF calculation
let chunkVectors = []; // TF-IDF representation for each chunk

// Initialize RAG from the static document database with adjustable sliders
function initializeRAG(sentenceCount = 2, overlap = 1) {
  allChunks = [];
  
  // 1. Chunk static documents
  documents.forEach(doc => {
    const chunks = chunkDocument(doc, sentenceCount, overlap);
    allChunks.push(...chunks);
  });
  
  // 2. Build index
  rebuildIndex();
}

// Injects dynamic telemetry data into RAG corpus with matching tuner parameters
function injectTelemetryChunk(telemetryText, sentenceCount = 2, overlap = 1) {
  // Remove any existing telemetry chunks from the list
  allChunks = allChunks.filter(chunk => chunk.docId !== 'telemetry');
  
  // Add new telemetry chunks
  const telemetryDoc = {
    id: 'telemetry',
    title: 'Live Simulator Flight Telemetry',
    category: 'Telemetry',
    content: telemetryText
  };
  
  const chunks = chunkDocument(telemetryDoc, sentenceCount, overlap);
  allChunks.push(...chunks);
  
  // Re-build terms weights
  rebuildIndex();
}

// Internal: Computes TF-IDF index for the active chunks
function rebuildIndex() {
  vocabulary = new Set();
  df = {};
  chunkVectors = [];
  
  // A. Tokenize and count Document Frequency (DF)
  const tokenizedChunks = allChunks.map(chunk => {
    const tokens = tokenize(chunk.content);
    const uniqueTokens = new Set(tokens);
    
    // Add to DF count
    uniqueTokens.forEach(token => {
      vocabulary.add(token);
      df[token] = (df[token] || 0) + 1;
    });
    
    return { chunk, tokens };
  });
  
  // B. Compute IDF for each term in vocabulary
  const N = allChunks.length;
  const idf = {};
  vocabulary.forEach(token => {
    // Standard IDF with smoothing to prevent division by zero
    idf[token] = Math.log(1 + N / (df[token] || 1)) + 1;
  });
  
  // C. Calculate TF-IDF vectors for each chunk
  tokenizedChunks.forEach(({ chunk, tokens }) => {
    // Term Frequency (TF) in this chunk
    const tf = {};
    tokens.forEach(token => {
      tf[token] = (tf[token] || 0) + 1;
    });
    
    // Normalize TF by document length
    const totalTerms = tokens.length || 1;
    const vector = {};
    
    vocabulary.forEach(token => {
      if (tf[token]) {
        const tfNorm = tf[token] / totalTerms;
        vector[token] = tfNorm * idf[token];
      } else {
        vector[token] = 0;
      }
    });
    
    // Calculate L2 Vector Norm (length) for cosine normalization
    let sumSq = 0;
    Object.values(vector).forEach(val => { sumSq += val * val; });
    const norm = Math.sqrt(sumSq) || 1;
    
    // Category anchors for 2D Projection visual map
    const anchors = {
      'History': { x: 80, y: 130 },
      'Specifications': { x: 80, y: 50 },
      'Physics': { x: 320, y: 130 },
      'Game Manual': { x: 320, y: 50 },
      'Telemetry': { x: 200, y: 90 }
    };
    
    const cat = chunk.category;
    const anchor = anchors[cat] || { x: 200, y: 90 };
    // Deterministic spread calculation
    let hash = 0;
    for (let charIdx = 0; charIdx < chunk.id.length; charIdx++) {
      hash = chunk.id.charCodeAt(charIdx) + ((hash << 5) - hash);
    }
    const offsetAngle = (Math.abs(hash) % 360) * Math.PI / 180;
    const offsetDist = 12 + (Math.abs(hash) % 18);
    
    chunkVectors.push({
      chunk,
      vector,
      norm,
      visualX: anchor.x + Math.cos(offsetAngle) * offsetDist,
      visualY: anchor.y + Math.sin(offsetAngle) * offsetDist
    });
  });
}

// Performs retrieval based on Cosine Similarity
function retrieve(queryText, topK = 3) {
  const queryTokens = tokenize(queryText);
  if (queryTokens.length === 0 || allChunks.length === 0) {
    return allChunks.slice(0, topK).map(chunk => ({ chunk, score: 0 }));
  }
  
  // A. Calculate Query Vector
  const queryTf = {};
  queryTokens.forEach(token => {
    queryTf[token] = (queryTf[token] || 0) + 1;
  });
  
  const totalQueryTerms = queryTokens.length;
  const queryVector = {};
  
  // Use same IDF as documents
  const N = allChunks.length;
  vocabulary.forEach(token => {
    if (queryTf[token]) {
      const qTfNorm = queryTf[token] / totalQueryTerms;
      const tokenDf = df[token] || 0;
      const tokenIdf = Math.log(1 + N / (tokenDf || 1)) + 1;
      queryVector[token] = qTfNorm * tokenIdf;
    } else {
      queryVector[token] = 0;
    }
  });
  
  // Calculate L2 Vector Norm for Query
  let querySumSq = 0;
  Object.values(queryVector).forEach(val => { querySumSq += val * val; });
  const queryNorm = Math.sqrt(querySumSq) || 1;
  
  // B. Calculate Cosine Similarity with all chunks
  const scores = chunkVectors.map(item => {
    let dotProduct = 0;
    // We only need to dot product where query terms are non-zero
    queryTokens.forEach(token => {
      if (item.vector[token]) {
        dotProduct += queryVector[token] * item.vector[token];
      }
    });
    
    const score = dotProduct / (queryNorm * item.norm);
    
    return {
      chunk: item.chunk,
      score: isNaN(score) ? 0 : score,
      visualX: item.visualX,
      visualY: item.visualY
    };
  });
  
  // C. Sort and slice
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK);
}

// Retrieval Inspector Details
function getRAGState() {
  return {
    chunks: allChunks,
    vocabSize: vocabulary.size,
    idfWeights: df,
    chunkVectors: chunkVectors.map(item => ({
      chunkId: item.chunk.id,
      docTitle: item.chunk.docTitle,
      category: item.chunk.category,
      content: item.chunk.content,
      visualX: item.visualX,
      visualY: item.visualY,
      terms: Object.entries(item.vector)
        .filter(([_, val]) => val > 0)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5) // top 5 weighted terms
        .map(([term, weight]) => ({ term, weight }))
    }))
  };
}

// Attach to window object for global scope
window.tokenize = tokenize;
window.chunkDocument = chunkDocument;
window.initializeRAG = initializeRAG;
window.injectTelemetryChunk = injectTelemetryChunk;
window.retrieve = retrieve;
window.getRAGState = getRAGState;

