import React from "react";
import { CodeBlock } from "./CodeBlock";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "../ui/Table";
import { Badge } from "../ui/Badge";
import type { EndpointSpec } from "../../data/apiDocsData";
import { cn } from "../../lib/utils";

interface EndpointCardProps {
  endpoint: EndpointSpec;
}

export const EndpointCard: React.FC<EndpointCardProps> = ({ endpoint }) => {
  return (
    <article
      id={endpoint.id}
      className="scroll-mt-24 pt-6 border-t border-border-subtle first:border-t-0 first:pt-0"
    >
      {/* Endpoint Title & Identity */}
      <div className="flex flex-col gap-2 mb-5">
        <div className="flex items-center gap-2.5">
          {/* Neutral Tone Method Badge */}
          <span
            className={cn(
              "font-mono text-xs font-bold px-2 py-0.5 rounded-[4px] uppercase border",
              endpoint.method === "POST"
                ? "bg-subtle text-text-primary border-border-default"
                : "bg-subtle text-text-secondary border-border-subtle"
            )}
          >
            {endpoint.method}
          </span>
          <h3 className="font-mono text-base sm:text-lg font-semibold text-text-primary">
            {endpoint.path}
          </h3>
        </div>

        <p className="text-sm font-medium text-text-primary">
          {endpoint.summary}
        </p>
        <p className="text-xs text-text-secondary leading-relaxed max-w-3xl">
          {endpoint.description}
        </p>
      </div>

      {/* Two-Subcolumn Split: Spec Tables on Left, Code Panel on Right */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        {/* Left Subcolumn (7 cols): Parameters, Response Schema, Error Codes */}
        <div className="xl:col-span-7 flex flex-col gap-6 min-w-0">
          {/* Parameters Table */}
          <div className="flex flex-col gap-2">
            <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
              Request Parameters
            </h4>
            {endpoint.parameters.length > 0 ? (
              <div className="overflow-x-auto">
                <Table dense>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-32">Parameter</TableHead>
                      <TableHead className="w-20">Type</TableHead>
                      <TableHead className="w-24">Required</TableHead>
                      <TableHead>Description</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {endpoint.parameters.map((p) => (
                      <TableRow key={p.name}>
                        <TableCell className="font-mono text-xs font-semibold text-text-primary">
                          {p.name}
                        </TableCell>
                        <TableCell className="font-mono text-[11px] text-text-secondary">
                          {p.type}
                        </TableCell>
                        <TableCell>
                          {p.required ? (
                            <Badge variant="brand" className="text-[10px] py-0">
                              Required
                            </Badge>
                          ) : (
                            <span className="font-mono text-[11px] text-text-muted">
                              {p.default ? `default: ${p.default}` : "optional"}
                            </span>
                          )}
                        </TableCell>
                        <TableCell className="text-xs text-text-secondary leading-normal">
                          {p.description}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <div className="p-3 rounded-control bg-subtle border border-border-subtle text-xs text-text-muted font-mono">
                No query parameters or request body required.
              </div>
            )}
          </div>

          {/* Response Schema Table */}
          <div className="flex flex-col gap-2">
            <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
              Response Fields
            </h4>
            <div className="overflow-x-auto">
              <Table dense>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-40">Field</TableHead>
                    <TableHead className="w-24">Type</TableHead>
                    <TableHead>Description</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {endpoint.responseFields.map((f) => (
                    <TableRow key={f.field}>
                      <TableCell className="font-mono text-xs font-semibold text-text-primary">
                        {f.field}
                      </TableCell>
                      <TableCell className="font-mono text-[11px] text-text-secondary">
                        {f.type}
                      </TableCell>
                      <TableCell className="text-xs text-text-secondary leading-normal">
                        {f.description}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Error Codes Table */}
          <div className="flex flex-col gap-2">
            <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
              Status Codes
            </h4>
            <div className="overflow-x-auto">
              <Table dense>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-20">Code</TableHead>
                    <TableHead className="w-36">Status</TableHead>
                    <TableHead>Reason</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell className="font-mono text-xs font-semibold text-brand">
                      200
                    </TableCell>
                    <TableCell className="font-medium text-text-primary text-xs">
                      OK
                    </TableCell>
                    <TableCell className="text-xs text-text-secondary">
                      Successful execution. Returns requested payload.
                    </TableCell>
                  </TableRow>
                  {endpoint.errorCodes.map((err) => (
                    <TableRow key={err.status}>
                      <TableCell className="font-mono text-xs font-semibold text-text-muted">
                        {err.status}
                      </TableCell>
                      <TableCell className="font-medium text-text-secondary text-xs">
                        {err.name}
                      </TableCell>
                      <TableCell className="text-xs text-text-secondary">
                        {err.description}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </div>

        {/* Right Subcolumn (5 cols): Sticky Code Snippets & Example Response */}
        <div className="xl:col-span-5 xl:sticky xl:top-20">
          <CodeBlock
            title={`${endpoint.method} ${endpoint.path}`}
            snippets={endpoint.snippets}
            exampleResponse={endpoint.exampleResponse}
          />
        </div>
      </div>
    </article>
  );
};

