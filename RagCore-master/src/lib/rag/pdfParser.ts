import * as pdfjsLib from 'pdfjs-dist';
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Set worker source using Vite's URL import
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

export interface ParsedPage {
  pageNumber: number;
  text: string;
  image?: string; // Base64 encoded page image
}

export interface ParsedPDF {
  pages: ParsedPage[];
  totalPages: number;
  metadata?: {
    title?: string;
    author?: string;
  };
}

export async function parsePDF(
  file: File,
  onProgress?: (current: number, total: number) => void
): Promise<ParsedPDF> {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  const pages: ParsedPage[] = [];
  const totalPages = pdf.numPages;

  for (let i = 1; i <= totalPages; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();

    const text = textContent.items
      .map((item: any) => item.str)
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

    // Only render the first page as a preview image to avoid huge CPU & RAM lag on big PDFs
    const image = i === 1 ? await renderPageToImage(page) : undefined;

    pages.push({
      pageNumber: i,
      text,
      image,
    });

    if (onProgress) {
      onProgress(i, totalPages);
    }
  }

  // Try to get metadata
  let metadata: ParsedPDF['metadata'];
  try {
    const meta = await pdf.getMetadata();
    metadata = {
      title: (meta.info as any)?.Title,
      author: (meta.info as any)?.Author,
    };
  } catch {
    // Metadata not available
  }

  return {
    pages,
    totalPages,
    metadata,
  };
}

async function renderPageToImage(page: any): Promise<string> {
  const viewport = page.getViewport({ scale: 1.0 });
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  if (!context) return '';

  // Set dimensions for a reasonable low-res preview (e.g., max 1024px)
  const MAX_DIM = 1024;
  const scale = Math.min(MAX_DIM / viewport.width, MAX_DIM / viewport.height, 1.0);
  const scaledViewport = page.getViewport({ scale });

  canvas.width = scaledViewport.width;
  canvas.height = scaledViewport.height;

  await page.render({
    canvasContext: context,
    viewport: scaledViewport,
  }).promise;

  return canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
}
