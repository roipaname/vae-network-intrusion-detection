import { useEffect, useRef, useState } from 'react';
import { api } from './api';

// Generic data-fetching hook. Pass a fetcher function and its dependency
// array; call `reload()` to refetch (e.g. after a user action).
//
// The loading spinner only ever appears before the *first* successful
// fetch this hook instance has seen. Every refetch after that -- a polling
// tick, a dataset/variant switch, a manual reload() -- keeps rendering the
// previous data in place and swaps it out silently once the new response
// arrives, so the UI never blanks out or flashes "Loading…" on its own.
export function useApi(fetcher, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    if (!hasLoadedRef.current) setLoading(true);
    fetcher()
      .then((result) => {
        if (cancelled) return;
        hasLoadedRef.current = true;
        setData(result);
        setError(null);
      })
      .catch((err) => {
        if (!cancelled) setError(err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, reloadKey]);

  return { data, loading, error, reload: () => setReloadKey((k) => k + 1) };
}

// Same as useApi, but refetches on an interval -- used for the live
// traffic/alerts feeds.
export function usePolling(fetcher, deps = [], intervalMs = 4000) {
  const result = useApi(fetcher, deps);
  useEffect(() => {
    const id = setInterval(result.reload, intervalMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, intervalMs]);
  return result;
}

// Tracks whether the backend is actually reachable, for the top bar's live
// status indicator.
export function useBackendStatus(intervalMs = 8000) {
  const [online, setOnline] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const check = () => {
      api
        .health()
        .then(() => {
          if (!cancelled) setOnline(true);
        })
        .catch(() => {
          if (!cancelled) setOnline(false);
        });
    };
    check();
    const id = setInterval(check, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return online;
}
