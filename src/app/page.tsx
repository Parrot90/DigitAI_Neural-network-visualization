import DigitAiClient from "@/components/digit-ai/digit-ai-client";
import ClientOnly from "@/components/client-only";
import { Skeleton } from "@/components/ui/skeleton";

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <ClientOnly fallback={
        <div className="w-full h-screen flex items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <Skeleton className="h-16 w-16 rounded-full" />
            <Skeleton className="h-8 w-64" />
            <Skeleton className="h-4 w-48" />
          </div>
        </div>
      }>
        <DigitAiClient />
      </ClientOnly>
    </main>
  );
}