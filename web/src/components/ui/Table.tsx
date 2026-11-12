import React from "react";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { cn } from "../../lib/utils";

export const Table = React.forwardRef<
  HTMLTableElement,
  React.TableHTMLAttributes<HTMLTableElement> & { dense?: boolean }
>(({ className, dense = false, ...props }, ref) => (
  <div className="relative w-full overflow-auto rounded-panel border border-border-subtle bg-card">
    <table
      ref={ref}
      className={cn(
        "w-full caption-bottom text-xs text-left",
        dense ? "text-xs" : "text-sm",
        className
      )}
      {...props}
    />
  </div>
));
Table.displayName = "Table";

export const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead
    ref={ref}
    className={cn(
      "border-b border-border-subtle bg-subtle text-[11px] font-semibold text-text-tertiary uppercase tracking-wider",
      className
    )}
    {...props}
  />
));
TableHeader.displayName = "TableHeader";

export const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn("divide-y divide-border-subtle/70", className)}
    {...props}
  />
));
TableBody.displayName = "TableBody";

export const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "transition-colors hover:bg-elevated/50 data-[state=selected]:bg-brand-subtle",
      className
    )}
    {...props}
  />
));
TableRow.displayName = "TableRow";

export interface TableHeadProps
  extends React.ThHTMLAttributes<HTMLTableCellElement> {
  sortable?: boolean;
  sortDirection?: "asc" | "desc" | false;
  onSort?: () => void;
}

export const TableHead = React.forwardRef<
  HTMLTableCellElement,
  TableHeadProps
>(
  (
    { className, sortable, sortDirection, onSort, children, ...props },
    ref
  ) => (
    <th
      ref={ref}
      className={cn(
        "h-9 px-3.5 text-left align-middle font-medium select-none",
        sortable && "cursor-pointer hover:text-text-primary transition-colors",
        className
      )}
      onClick={sortable ? onSort : undefined}
      {...props}
    >
      <div className="flex items-center gap-1.5">
        <span>{children}</span>
        {sortable && (
          <span className="text-text-tertiary">
            {sortDirection === "asc" && <ArrowUp className="w-3 h-3 text-brand" />}
            {sortDirection === "desc" && <ArrowDown className="w-3 h-3 text-brand" />}
            {!sortDirection && <ArrowUpDown className="w-3 h-3 text-text-tertiary opacity-70" />}
          </span>
        )}
      </div>
    </th>
  )
);
TableHead.displayName = "TableHead";

export const TableCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement> & { mono?: boolean }
>(({ className, mono, ...props }, ref) => (
  <td
    ref={ref}
    className={cn(
      "p-3 align-middle text-text-primary",
      mono && "font-mono tabular-nums text-xs",
      className
    )}
    {...props}
  />
));
TableCell.displayName = "TableCell";

