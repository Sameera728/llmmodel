import React, { useState, useEffect } from "react";
import { Upload, Link, Trash2, Database, FileText, CheckCircle, HelpCircle, HardDrive, Cpu, AlertTriangle } from "lucide-react";
import { IngestedDocument } from "../types";

interface DatasetRAGProps {
  onNotify: (msg: string) => void;
}

export default function DatasetRAG({ onNotify }: DatasetRAGProps) {
  const [documents, setDocuments] = useState<IngestedDocument[]>([]);
  const [webUrl, setWebUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      if (data.success) {
        setDocuments(data.documents);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleUploadSimulate = async (filename: string, size: number) => {
    setUploading(true);
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, size }),
      });
      const data = await res.json();
      if (data.success) {
        onNotify(`Successfully indexed ${filename}`);
        fetchDocuments();
      }
    } catch (e) {
      onNotify("Ingestion failure");
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      handleUploadSimulate(file.name, file.size);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleUploadSimulate(file.name, file.size);
    }
  };

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!webUrl.trim()) return;
    try {
      const parsedUrl = new URL(webUrl);
      const host = parsedUrl.hostname || "scraped_website";
      const displayFilename = `url_context_${host.replace(/\./g, "_")}.html`;
      handleUploadSimulate(displayFilename, 45200);
      setWebUrl("");
    } catch (err) {
      onNotify("Invalid URL format");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      const res = await fetch(`/api/documents/${id}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (data.success) {
        onNotify("Deleted vector fragments from index");
        fetchDocuments();
      }
    } catch (e) {
      onNotify("Failed to delete document");
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">
      {/* Upload Zone */}
      <div className="lg:col-span-2 space-y-6">
        <div className="bg-zinc-900/40 backdrop-blur-md rounded-xl border border-zinc-800 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Database size={16} className="text-blue-500" />
            <h3 className="text-sm font-semibold text-zinc-100">RAG Document Ingestor</h3>
          </div>

          <p className="text-xs text-zinc-400 mb-5 leading-relaxed">
            Upload custom domain documentation (PDFs, DOCX, Markdown, or HTML urls). The pipeline parses text via PyPDF/Docx wrappers, divides sentences using overlapping sliding windows, embeds blocks via SentenceTransformers, and saves indices to local FAISS indices.
          </p>

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-all cursor-pointer h-56 ${
              isDragActive
                ? "border-blue-500 bg-blue-500/5"
                : "border-zinc-800 bg-[#09090b]/45 hover:border-zinc-700/80"
            }`}
            onClick={() => document.getElementById("file-upload")?.click()}
          >
            <input
              id="file-upload"
              type="file"
              accept=".pdf,.docx,.txt,.md,.html"
              onChange={handleFileChange}
              className="hidden"
            />
            {uploading ? (
              <div className="flex flex-col items-center gap-3">
                <span className="w-8 h-8 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></span>
                <span className="text-xs text-zinc-300 font-mono">Running sentence tokenization...</span>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 text-center">
                <div className="w-12 h-12 rounded-full bg-zinc-900/60 flex items-center justify-center border border-zinc-800 shadow-inner">
                  <Upload size={20} className="text-zinc-400" />
                </div>
                <div>
                  <p className="text-xs text-zinc-200 font-medium">Drag & drop files or click to select</p>
                  <p className="text-[10px] text-zinc-500 mt-1 font-mono">PDF, DOCX, TXT, MD, HTML (Max 50MB)</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Website Scraping */}
        <div className="bg-zinc-900/40 backdrop-blur-md rounded-xl border border-zinc-800 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Link size={16} className="text-blue-500" />
            <h3 className="text-sm font-semibold text-zinc-100">Live Website Crawling</h3>
          </div>

          <form onSubmit={handleUrlSubmit} className="flex gap-2">
            <input
              type="url"
              value={webUrl}
              onChange={(e) => setWebUrl(e.target.value)}
              placeholder="https://example.com/documentation/api"
              className="flex-1 bg-[#09090b] text-xs text-zinc-300 border border-zinc-800 rounded-lg px-3.5 py-2 focus:border-blue-500/50 outline-none"
            />
            <button
              type="submit"
              disabled={uploading || !webUrl.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-800 text-white disabled:text-zinc-500 font-semibold px-4 py-2 rounded-lg text-xs transition-colors cursor-pointer"
            >
              Scrape Site
            </button>
          </form>
          <p className="text-[10px] text-zinc-500 font-mono mt-2.5">
            Parses webpage HTML, purges boilerplate javascript & stylesheets using BeautifulSoup, and indexes content.
          </p>
        </div>
      </div>

      {/* Database Status & Index Manager */}
      <div className="bg-zinc-900/40 backdrop-blur-md rounded-xl border border-zinc-800 p-6 flex flex-col h-full">
        <div className="flex items-center justify-between pb-3 mb-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <HardDrive size={16} className="text-blue-500" />
            <h3 className="text-sm font-semibold text-zinc-100">FAISS Index Registry</h3>
          </div>
          <span className="px-2 py-0.5 rounded bg-zinc-800 border border-zinc-700/60 text-[9px] text-zinc-400 font-mono">
            PERSISTED
          </span>
        </div>

        {/* Index File Information */}
        <div className="grid grid-cols-2 gap-3 mb-4 text-xs font-mono">
          <div className="bg-[#0c0c0e] border border-zinc-800/50 rounded-lg p-3">
            <span className="text-[10px] text-zinc-500 block">Total Chunks</span>
            <span className="text-blue-400 font-bold text-base">
              {documents.reduce((acc, curr) => acc + curr.chunks, 0)}
            </span>
          </div>
          <div className="bg-[#0c0c0e] border border-zinc-800/50 rounded-lg p-3">
            <span className="text-[10px] text-zinc-500 block">Index Volume</span>
            <span className="text-emerald-500 font-bold text-base">
              {formatSize(documents.reduce((acc, curr) => acc + curr.size, 0))}
            </span>
          </div>
        </div>

        {/* Document list */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar max-h-72">
          {documents.length > 0 ? (
            documents.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center justify-between p-3 bg-[#09090b]/45 rounded-lg border border-zinc-800/50 hover:border-zinc-700/60 transition-colors"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <FileText size={16} className="text-blue-400 shrink-0" />
                  <div className="min-w-0 text-xs">
                    <p className="font-semibold text-zinc-200 truncate">{doc.name}</p>
                    <p className="text-[10px] text-zinc-500 mt-0.5 font-mono">
                      {formatSize(doc.size)} | {doc.chunks} vectors
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => handleDelete(doc.id)}
                  className="p-1 rounded-md text-zinc-500 hover:text-red-400 hover:bg-zinc-800/50 transition-colors"
                  title="Remove from FAISS index"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center py-8 text-zinc-500">
              <AlertTriangle size={18} className="text-blue-500/60 mb-2" />
              <p className="text-xs">No vector data loaded</p>
              <p className="text-[10px] text-zinc-600 mt-1">Upload a source doc to instantiate RAG contexts</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
