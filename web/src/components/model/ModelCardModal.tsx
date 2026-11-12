import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/Dialog";
import { Badge } from "../ui/Badge";

interface ModelCardModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const ModelCardModal: React.FC<ModelCardModalProps> = ({
  open,
  onOpenChange,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <DialogTitle>SSD MobileNet V2 Model Card</DialogTitle>
            <Badge variant="neutral">v1.0.0</Badge>
          </div>
          <DialogDescription>
            Comprehensive architecture, training provenance, and operational boundaries.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-5 text-xs text-text-secondary mt-2">
          {/* Section 1 */}
          <div className="space-y-1.5">
            <h4 className="text-sm font-semibold text-text-primary">1. Model Overview</h4>
            <div className="grid grid-cols-2 gap-2 rounded-control bg-subtle p-3 font-mono text-[11px]">
              <div><span className="text-text-muted">Task:</span> 2D Object Detection</div>
              <div><span className="text-text-muted">Framework:</span> PyTorch 2.x (native)</div>
              <div><span className="text-text-muted">Input Size:</span> 300 × 300 RGB</div>
              <div><span className="text-text-muted">Anchors:</span> 2,268 prior boxes</div>
              <div><span className="text-text-muted">Classes:</span> 20 targets + background</div>
              <div><span className="text-text-muted">Latency:</span> ~15 ms CPU (p95 &lt; 300 ms)</div>
            </div>
          </div>

          {/* Section 2 */}
          <div className="space-y-1.5">
            <h4 className="text-sm font-semibold text-text-primary">2. Architecture & Heads</h4>
            <p className="leading-relaxed">
              The detector uses a MobileNet V2 inverted residual backbone with linear bottlenecks and depthwise separable convolutions, implemented directly in native PyTorch without external computer vision library dependencies.
            </p>
            <p className="leading-relaxed">
              Six SSD detection layers predict offsets and class distributions at strides 8, 16, 32, 64, 100, and 300. Multi-scale feature maps assign aspect-ratio anchors (1:1, 2:1, 1:2, 3:1, 1:3) to match varying object scales. Post-processing combines a 0.25 confidence threshold with class-agnostic Non-Maximum Suppression (IoU ≥ 0.45).
            </p>
          </div>

          {/* Section 3 */}
          <div className="space-y-1.5">
            <h4 className="text-sm font-semibold text-text-primary">3. Training & Curriculum</h4>
            <p className="leading-relaxed">
              The model was initialized from COCO pre-trained weights and fine-tuned on a 20-class Open Images subset (10 COCO-shared classes and 10 novel target categories). Training used a two-phase transfer learning curriculum:
            </p>
            <ul className="list-disc list-inside space-y-1 pl-1 text-[11px]">
              <li>
                <strong className="text-text-primary">Phase 1 (2 epochs):</strong> Backbone frozen; only classification and regression output heads trained with LR 1.0 × 10⁻³.
              </li>
              <li>
                <strong className="text-text-primary">Phase 2 (3 epochs):</strong> Full network unfrozen; fine-tuned end-to-end with cosine annealing LR 1.0 × 10⁻⁴.
              </li>
            </ul>
          </div>

          {/* Section 4 */}
          <div className="space-y-1.5">
            <h4 className="text-sm font-semibold text-text-primary">4. Operational Boundaries & Known Limits</h4>
            <div className="space-y-2 rounded-control border border-border-subtle bg-subtle p-3 text-[11px]">
              <div>
                <strong className="text-text-primary block">Small Object Degradation:</strong>
                Objects occupying fewer than 32 × 32 pixels suffer reduced recall due to 300 × 300 input downsampling.
              </div>
              <div>
                <strong className="text-text-primary block">Dense Overlapping Objects:</strong>
                Scenes containing more than 5 intersecting instances within a small radius may encounter bounding box suppression or localization jitter.
              </div>
              <div>
                <strong className="text-text-primary block">Low-Light & Motion Blur:</strong>
                Depthwise convolutions rely heavily on edge contrast; low-contrast or blurred imagery leads to false negatives.
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
