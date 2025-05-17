import React, { useState, useEffect, useContext, useRef, useMemo } from 'react';
import { Clock, Search, Bookmark, Share2, Copy, ChevronDown, ChevronUp } from 'lucide-react';
import { ThemeContext } from '../styles/theme';
/**
 * TranscriptionDisplay component for Luna AI
 * Displays video transcriptions with interactive timestamps and search functionality
 */
const TranscriptionDisplay = ({ 
  transcription, 
  onTimestampClick, 
  isLoading = false,
  currentTime = 0,
  speakers = []
}) => {
  const { darkMode } = useContext(ThemeContext);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredTranscription, setFilteredTranscription] = useState([]);
  const [expandedSections, setExpandedSections] = useState({});
  const [selectedSpeaker, setSelectedSpeaker] = useState('all');
  const containerRef = useRef(null);
  const [activeSegment, setActiveSegment] = useState(null);
  
  // Debug logging to check incoming transcription data
  useEffect(() => {
    console.log("TranscriptionDisplay received data:", { 
      transcription, 
      type: typeof transcription, 
      isLoading,
      hasData: transcription && (typeof transcription === 'string' || Array.isArray(transcription))
    });
  }, [transcription, isLoading]);
  
  // Format transcript segments properly if not already formatted
  const segments = useMemo(() => {
    if (!transcription) return [];
    
    // Handle object format (which might come from Socket.IO JSON)
    if (typeof transcription === 'object' && !Array.isArray(transcription)) {
      // If it has a 'segments' or 'data' property
      if (transcription.segments) return transcription.segments;
      if (transcription.data) return transcription.data;
      
      // Check if it's a non-array object that might be a single segment
      if (transcription.text) return [transcription];
    }
    
    if (Array.isArray(transcription)) return transcription;
    if (typeof transcription === 'string') return parseTranscription(transcription);
    
    console.warn("Unknown transcription format:", transcription);
    return [];
  }, [transcription]);
  
  // Group transcription segments by minute for better organization
  const groupedSegments = useMemo(() => {
    return groupSegmentsByMinute(segments);
  }, [segments]);
  
  // Filter transcription based on search query and selected speaker
  useEffect(() => {
    if (!segments.length) {
      setFilteredTranscription([]);
      return;
    }
    
    let filtered = [...segments];
    
    // Filter by search query if it exists
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(segment => 
        segment.text && segment.text.toLowerCase().includes(query)
      );
    }
    
    // Filter by selected speaker if not "all"
    if (selectedSpeaker !== 'all') {
      filtered = filtered.filter(segment => 
        segment.speaker === selectedSpeaker
      );
    }
    
    setFilteredTranscription(filtered);
  }, [searchQuery, segments, selectedSpeaker]);
  
  // Find and set the active segment based on current playback time
  useEffect(() => {
    if (!segments.length || currentTime === undefined) return;
    
    const current = segments.findIndex((segment, index) => {
      if (!segment.start && segment.start !== 0) return false;
      
      const nextSegment = segments[index + 1];
      if (!nextSegment) return true;
      
      return currentTime >= segment.start && currentTime < nextSegment.start;
    });
    
    if (current !== -1 && current !== activeSegment) {
      setActiveSegment(current);
      
      // Auto-scroll to active segment - use a safer ID format
      if (containerRef.current && segments[current]) {
        // Convert to string and replace decimal points to ensure valid ID
        const safeId = `segment-${String(segments[current].start).replace('.', '-')}`;
        const segmentEl = document.getElementById(safeId);
        
        if (segmentEl) {
          containerRef.current.scrollTop = segmentEl.offsetTop - 100;
        }
      }
    }
  }, [currentTime, segments, activeSegment]);
  
  // Handle timestamp click
  const handleTimestampClick = (time) => {
    if (onTimestampClick && typeof time === 'number') {
      onTimestampClick(time);
    }
  };
  
  // Toggle section expansion
  const toggleSection = (minute) => {
    setExpandedSections(prev => ({
      ...prev,
      [minute]: !prev[minute]
    }));
  };
  
  // Format time (seconds) to MM:SS
  const formatTime = (seconds) => {
    if (seconds === undefined || seconds === null) return '00:00';
    
    const mins = Math.floor(Math.max(0, seconds) / 60);
    const secs = Math.floor(Math.max(0, seconds) % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Handle copy to clipboard
  const copyToClipboard = () => {
    try {
      const text = segments.map(segment => segment.text || '').join(' ');
      navigator.clipboard.writeText(text);
      // You would typically show a toast notification here
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };
  
  // Function to parse raw transcription text if needed
  function parseTranscription(text) {
    // This is a simplified parser, enhance based on your actual data format
    if (!text) return [];
    
    try {
      // First check if it's JSON
      try {
        const jsonData = JSON.parse(text);
        if (Array.isArray(jsonData)) {
          return jsonData.map((item, idx) => ({
            id: item.id || idx,
            start: Number(item.start) || 0,
            end: Number(item.end) || (Number(item.start) + 5),
            text: item.text || '',
            speaker: item.speaker || 'Speaker'
          }));
        }
        
        // If it's an object with segments
        if (jsonData.segments) return jsonData.segments;
        if (jsonData.data) return jsonData.data;
      } catch (e) {
        // Not JSON, continue with text parsing
        console.log("Not JSON, trying text parsing", e);
      }
      
      // Simple parsing assuming format of "00:00 Speaker: Text" per line
      return text.split('\n')
        .filter(line => line.trim())
        .map((line, idx) => {
          // More robust regex patterns
          const timeMatch = line.match(/^(\d{1,2}:\d{1,2}(?:\.\d+)?)/);
          const speakerMatch = line.match(/^\s*(?:\d{1,2}:\d{1,2}(?:\.\d+)?)\s+([^:]+):/);
          
          let start = 0;
          if (timeMatch) {
            const timeParts = timeMatch[1].split(':');
            start = parseInt(timeParts[0], 10) * 60 + parseFloat(timeParts[1] || 0);
          }
          
          let processedText = line;
          if (timeMatch) {
            processedText = line.substring(timeMatch[0].length).trim();
          }
          
          if (speakerMatch) {
            processedText = line.replace(/^\s*(?:\d{1,2}:\d{1,2}(?:\.\d+)?)\s+([^:]+):/, '').trim();
          }
          
          return {
            id: idx,
            start,
            end: start + 5, // Approximate end time
            text: processedText,
            speaker: speakerMatch ? speakerMatch[1].trim() : 'Speaker'
          };
        });
    } catch (error) {
      console.error('Error parsing transcription:', error);
      return [];
    }
  }
  
  // Group segments by minute for better UI organization
  function groupSegmentsByMinute(segments) {
    if (!segments || !Array.isArray(segments)) return {};
    
    const grouped = {};
    
    segments.forEach(segment => {
      if (segment && typeof segment.start === 'number') {
        const minute = Math.floor(segment.start / 60);
        if (!grouped[minute]) {
          grouped[minute] = [];
        }
        grouped[minute].push(segment);
      }
    });
    
    return grouped;
  }
  
  // Create a safe ID from segment start time
  const getSafeId = (startTime) => {
    if (startTime === undefined || startTime === null) return 'segment-unknown';
    return `segment-${String(startTime).replace('.', '-')}`;
  };
  
  // Debug indicator for development
  const showDebugInfo = () => {
    if (!segments.length && !isLoading && transcription) {
      return (
        <div className="p-2 bg-red-100 text-red-800 rounded mb-2">
          <p>Debug: Received transcription data but couldn't process it</p>
          <p>Type: {typeof transcription}</p>
          {typeof transcription === 'string' && <p>Length: {transcription.length} chars</p>}
          {Array.isArray(transcription) && <p>Items: {transcription.length}</p>}
        </div>
      );
    }
    return null;
  };
  
  return (
    <div className={`flex flex-col h-full rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-md overflow-hidden transition-colors duration-200`}>
      {/* Transcript Header */}
      <div className={`p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} flex justify-between items-center`}>
        <div className="flex items-center">
          <Clock size={18} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
          <h2 className="ml-2 font-medium">Transcript</h2>
          {isLoading && (
            <div className="ml-3 flex items-center">
              <div className={`animate-pulse h-2 w-16 rounded ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`}></div>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button 
            className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
            title="Copy transcript"
            onClick={copyToClipboard}
          >
            <Copy size={16} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
          </button>
          <button 
            className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
            title="Share transcript"
          >
            <Share2 size={16} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
          </button>
          <button 
            className={`p-1.5 rounded-full ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
            title="Bookmark"
          >
            <Bookmark size={16} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
          </button>
        </div>
      </div>
      
      {/* Search and Filter Bar */}
      <div className={`px-4 py-3 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} space-y-2`}>
        <div className={`flex items-center rounded-md overflow-hidden ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
          <Search size={18} className={`mx-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search transcript..."
            className={`flex-1 py-2 px-2 outline-none ${darkMode ? 'bg-gray-700 text-white placeholder:text-gray-400' : 'bg-gray-100 text-gray-900 placeholder:text-gray-500'}`}
          />
          {searchQuery && (
            <button 
              onClick={() => setSearchQuery('')}
              className={`mr-2 ${darkMode ? 'text-gray-400 hover:text-gray-300' : 'text-gray-500 hover:text-gray-700'}`}
            >
              ✕
            </button>
          )}
        </div>
        
        {speakers.length > 1 && (
          <div className="flex items-center gap-2 overflow-x-auto no-scrollbar py-1">
            <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Filter:</span>
            <button
              onClick={() => setSelectedSpeaker('all')}
              className={`px-2 py-0.5 text-sm rounded-full whitespace-nowrap ${
                selectedSpeaker === 'all' 
                  ? (darkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-800') 
                  : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700')
              }`}
            >
              All speakers
            </button>
            {speakers.map(speaker => (
              <button
                key={speaker}
                onClick={() => setSelectedSpeaker(speaker)}
                className={`px-2 py-0.5 text-sm rounded-full whitespace-nowrap ${
                  selectedSpeaker === speaker 
                    ? (darkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-800') 
                    : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700')
                }`}
              >
                {speaker}
              </button>
            ))}
          </div>
        )}
      </div>
      
      {/* Debug Info */}
      {showDebugInfo()}
      
      {/* Transcription Content */}
      <div 
        ref={containerRef}
        className={`flex-1 overflow-y-auto px-4 py-2 ${darkMode ? 'scrollbar-dark' : 'scrollbar-light'}`}
      >
        {isLoading ? (
          // Loading state
          <div className="space-y-4 py-4">
            {[...Array(8)].map((_, idx) => (
              <div key={idx} className="flex animate-pulse">
                <div className={`w-12 flex-shrink-0 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} h-5 rounded mr-3`}></div>
                <div className="flex-1 space-y-2">
                  <div className={`h-4 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} rounded w-3/4`}></div>
                  <div className={`h-4 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} rounded w-1/2`}></div>
                </div>
              </div>
            ))}
          </div>
        ) : searchQuery ? (
          // Search results
          <div className="py-2">
            <p className={`text-sm mb-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {filteredTranscription.length} {filteredTranscription.length === 1 ? 'result' : 'results'} for "{searchQuery}"
            </p>
            {filteredTranscription.length > 0 ? (
              <div className="space-y-4">
                {filteredTranscription.map((segment) => (
                  <div 
                    key={segment.id || String(segment.start)}
                    id={getSafeId(segment.start)}
                    className={`
                      group relative rounded-md p-2 cursor-pointer 
                      ${activeSegment === segments.indexOf(segment) ? 
                        (darkMode ? 'bg-indigo-900/30 border-l-2 border-indigo-500' : 'bg-indigo-50 border-l-2 border-indigo-500') : 
                        (darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-gray-50')
                      }
                    `}
                    onClick={() => handleTimestampClick(segment.start)}
                  >
                    <div className="flex items-start">
                      <span 
                        className={`inline-block w-14 flex-shrink-0 font-mono text-sm ${
                          darkMode ? 'text-indigo-400' : 'text-indigo-600'
                        }`}
                      >
                        {formatTime(segment.start)}
                      </span>
                      <div className="flex-1">
                        {segment.speaker && (
                          <span className={`font-medium ${darkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                            {segment.speaker}:
                          </span>
                        )}
                        <span className={`ml-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                          {highlightText(segment.text, searchQuery)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                No results found for "{searchQuery}"
              </div>
            )}
          </div>
        ) : !segments.length ? (
          // Empty state
          <div className={`flex flex-col items-center justify-center h-full text-center ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            <Clock size={36} className="mb-2 opacity-50" />
            <p className="text-sm mb-1">No transcription available yet</p>
            <p className="text-xs opacity-75">Upload a video or provide a YouTube URL to generate a transcript</p>
          </div>
        ) : (
          // Regular transcript display
          <div className="py-2 space-y-3">
            {Object.keys(groupedSegments).length > 0 ? (
              Object.keys(groupedSegments).map(minute => {
                const segments = groupedSegments[minute];
                if (!segments || !segments.length) return null;
                
                const firstSegment = segments[0];
                if (!firstSegment) return null;
                
                const isExpanded = expandedSections[minute] !== false; // Default to expanded
                
                return (
                  <div key={minute} className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    <div 
                      className={`
                        flex items-center cursor-pointer py-1 
                        ${darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-gray-100'}
                        rounded px-1 transition-colors duration-150
                      `}
                      onClick={() => toggleSection(minute)}
                    >
                      {isExpanded ? 
                        <ChevronUp size={16} className={darkMode ? 'text-gray-400' : 'text-gray-500'} /> : 
                        <ChevronDown size={16} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
                      }
                      <span 
                        className={`ml-1 text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}
                      >
                        {formatTime(firstSegment.start).split(':')[0]}:00 - {formatTime(firstSegment.start).split(':')[0]}:59
                      </span>
                    </div>
                    
                    {isExpanded && (
                      <div className="ml-6 space-y-3 mt-2">
                        {segments.map((segment) => (
                          <div 
                            key={segment.id || String(segment.start)}
                            id={getSafeId(segment.start)}
                            className={`
                              group relative rounded-md p-2 cursor-pointer transition-colors
                              ${segments.indexOf(segment) === activeSegment ? 
                                (darkMode ? 'bg-indigo-900/30 border-l-2 border-indigo-500' : 'bg-indigo-50 border-l-2 border-indigo-500') : 
                                (darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-gray-50')
                              }
                            `}
                            onClick={() => handleTimestampClick(segment.start)}
                          >
                            <div className="flex">
                              <span 
                                className={`inline-block w-14 flex-shrink-0 font-mono text-sm ${
                                  darkMode ? 'text-indigo-400' : 'text-indigo-600'
                                }`}
                              >
                                {formatTime(segment.start)}
                              </span>
                              <div className="flex-1">
                                {segment.speaker && (
                                  <span className={`font-medium ${darkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                                    {segment.speaker}:
                                  </span>
                                )}
                                <span className={`ml-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                  {segment.text || ''}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                The transcript appears to be in an unexpected format. Please check the transcription service.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Helper function to highlight search terms in text
function highlightText(text, query) {
  if (!query || !text) return text || '';
  
  try {
    const parts = text.split(new RegExp(`(${escapeRegExp(query)})`, 'gi'));
    return (
      <>
        {parts.map((part, i) => 
          part.toLowerCase() === query.toLowerCase() ? (
            <span key={i} className="bg-yellow-300 text-gray-800 rounded px-0.5">{part}</span>
          ) : (
            part
          )
        )}
      </>
    );
  } catch (error) {
    console.error('Error highlighting text:', error);
    return text || '';
  }
}

// Helper function to escape special characters in regex
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

export default TranscriptionDisplay;