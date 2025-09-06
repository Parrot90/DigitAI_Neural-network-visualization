'use client';

import { useEffect, useState } from 'react';

export default function HydrationDebug() {
  const [isClient, setIsClient] = useState(false);
  
  useEffect(() => {
    setIsClient(true);
    console.log('HydrationDebug: Client hydrated');
  }, []);

  return (
    <div className="fixed top-4 right-4 bg-red-500 text-white p-2 text-xs z-50">
      {isClient ? 'CLIENT' : 'SERVER'}
    </div>
  );
}
