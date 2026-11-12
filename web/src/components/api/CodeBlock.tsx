import React, { useState } from "react";
import { Copy, Check, Terminal } from "lucide-react";
import { cn } from "../../lib/utils";

export interface CodeBlockProps {
  title?: string;
  snippets: {
    curl: string;
    js: string;
    python: string;
  };
  exampleResponse?: string;
  className?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  title,
  snippets,
  exampleResponse,
  className,
}) => {
  const [activeTab, setActiveTab] = useState<"curl" | "js" | "python" | "response">("curl");
  const [copied, setCopied] = useState(false);

  const currentContent = activeTab === "response" && exampleResponse
    ? exampleResponse
    : snippets[activeTab as "curl" | "js" | "python"];

  const handleCopy = () => {
    void navigator.clipboard.writeText(currentContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={cn(
        "rounded-panel border border-border-subtle bg-card overflow-hidden flex flex-col shadow-sm",
        className
      )}
    >
      {/* Header with Title and Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-subtle px-3 py-2 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-text-tertiary" />
          <span className="text-xs font-medium text-text-secondary">
            {title || "Request Example"}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {/* Tabs */}
          <div className="flex items-center rounded-control bg-elevated p-0.5 border border-border-subtle text-[11px] font-mono">
            <button
              type="button"
              onClick={() => setActiveTab("curl")}
              className={cn(
                "px-2 py-0.5 rounded-[4px] transition-colors",
                activeTab === "curl"
                  ? "bg-card text-brand font-semibold shadow-sm"
                  : "text-text-muted hover:text-text-primary"
              )}
            >
              cURL
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("js")}
              className={cn(
                "px-2 py-0.5 rounded-[4px] transition-colors",
                activeTab === "js"
                  ? "bg-card text-brand font-semibold shadow-sm"
                  : "text-text-muted hover:text-text-primary"
              )}
            >
              Node.js
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("python")}
              className={cn(
                "px-2 py-0.5 rounded-[4px] transition-colors",
                activeTab === "python"
                  ? "bg-card text-brand font-semibold shadow-sm"
                  : "text-text-muted hover:text-text-primary"
              )}
            >
              Python
            </button>
            {exampleResponse && (
              <button
                type="button"
                onClick={() => setActiveTab("response")}
                className={cn(
                  "px-2 py-0.5 rounded-[4px] transition-colors border-l border-border-subtle/50 ml-0.5 pl-2",
                  activeTab === "response"
                    ? "bg-card text-brand font-semibold shadow-sm"
                    : "text-text-muted hover:text-text-primary"
                )}
              >
                200 OK
              </button>
            )}
          </div>

          {/* Copy Button */}
          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1 h-6 px-2 text-[11px] font-medium rounded-control bg-elevated hover:bg-subtle text-text-secondary hover:text-text-primary border border-border-subtle transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand"
            aria-label="Copy snippet to clipboard"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-status-success" />
                <span className="text-status-success font-mono">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3 text-text-tertiary" />
                <span className="hidden sm:inline">Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Code Area */}
      <pre className="p-3.5 text-xs font-mono text-text-primary bg-canvas overflow-x-auto leading-relaxed max-h-96 select-text">
        <code>{currentContent}</code>
      </pre>
    </div>
  );
};
