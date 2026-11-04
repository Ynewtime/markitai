using System.Collections.ObjectModel;

namespace Markitai.Core;

/// <summary>
/// The Convert list (native/DESIGN.md 4.1): newest first on screen, oldest first
/// to convert, at most two at a time. Every member runs on the owner's thread;
/// conversions run on the thread pool and come back through <c>post</c>.
/// </summary>
public sealed class ConversionQueue(IConverter converter, Func<string> outputFolder, Action<Action> post, TimeProvider clock)
{
    public const int Slots = 2;

    long nextOrder;

    // What the workers and Quit share, under one lock: the folders claimed for runs
    // still going, the claims in flight (a folder being made), and the results of runs
    // that ended but have not reached the owner's thread yet. Quit waits for claims in
    // flight (2 s at most), so every folder made before Quit is in `claimed` when Quit
    // deletes them, and a claim that ends after Quit deletes its own folder. A run that
    // ended before Quit keeps its folder and its result.
    readonly object gate = new();
    readonly Dictionary<ConversionItem, string> claimed = [];
    readonly Dictionary<ConversionItem, ItemState> ended = [];
    int inFlight;
    bool quitting;

    static readonly TimeSpan ClaimWait = TimeSpan.FromSeconds(2);

    public ObservableCollection<ConversionItem> Items { get; } = [];

    /// <summary>Raised when an item reaches Done or Failed.</summary>
    public event Action<ConversionItem>? Finished;

    /// <summary>Raised after any item's state changes.</summary>
    public event Action? StateChanged;

    public sealed record AddResult(IReadOnlyList<ConversionItem> Added, IReadOnlyList<Source> Duplicates);

    public int ConvertingCount => Items.Count(i => i.State is Converting);

    public int QueuedCount => Items.Count(i => i.State is Queued);

    public bool HasFinished => Items.Any(i => i.State is Done or Failed);

    /// <summary>Adds the sources in the order given, above the existing rows. A source already Queued or Converting is a duplicate.</summary>
    public AddResult Add(IEnumerable<Source> sources, ConversionOptions options)
    {
        var added = new List<ConversionItem>();
        var duplicates = new List<Source>();
        foreach (var source in sources)
        {
            if (Items.Any(i => i.State is Queued or Converting && i.Source.SameAs(source)) || added.Any(i => i.Source.SameAs(source)))
            {
                duplicates.Add(source);
                continue;
            }
            added.Add(new ConversionItem(source, options, clock.GetUtcNow(), nextOrder++));
        }
        for (var i = 0; i < added.Count; i++)
            Items.Insert(i, added[i]);
        Pump();
        return new(added, duplicates);
    }

    public bool Remove(ConversionItem item) => item.State is not Converting && Items.Remove(item);

    public void ClearFinished()
    {
        foreach (var item in Items.Where(i => i.State is Done or Failed).ToList())
            Items.Remove(item);
    }

    public void Retry(ConversionItem item, ConversionOptions options)
    {
        if (item.State is not Failed)
            return;
        item.Options = options;
        item.Order = nextOrder++;
        item.State = Queued.Instance;
        StateChanged?.Invoke();
        Pump();
    }

    void Pump()
    {
        while (!quitting && ConvertingCount < Slots && Items.Where(i => i.State is Queued).MinBy(i => i.Order) is { } next)
            Start(next);
    }

    void Start(ConversionItem item)
    {
        item.State = new Converting(clock.GetUtcNow());
        StateChanged?.Invoke();
        var (source, parent, options) = (item.Source, outputFolder(), item.Options);
        Task.Run(() =>
        {
            if (!BeginClaim())
                return;
            var claim = Conversion.Claim(converter, source, parent);
            if (!EndClaim(item, claim.Folder))
                return;
            var state = claim.Folder is { } folder ? Conversion.Convert(converter, source, folder, options) : claim.Failure!;
            if (state is Failed failed)
                state = failed with { Folder = parent };
            lock (gate)
            {
                if (quitting)
                    return;
                claimed.Remove(item);
                ended[item] = state;
            }
            post(() => Finish(item));
        });
    }

    bool BeginClaim()
    {
        lock (gate)
        {
            if (quitting)
                return false;
            inFlight++;
            return true;
        }
    }

    /// <summary>Keeps the folder for the run; false once quitting, after deleting it.</summary>
    bool EndClaim(ConversionItem item, string? folder)
    {
        bool kept;
        lock (gate)
        {
            kept = !quitting;
            if (kept && folder is not null)
                claimed[item] = folder;
        }
        // A claim that ends after Quit deletes its own folder before Quit stops waiting for it.
        if (!kept && folder is not null)
            Conversion.Delete(folder);
        lock (gate)
        {
            inFlight--;
            Monitor.PulseAll(gate);
        }
        return kept;
    }

    /// <summary>
    /// The user confirmed Quit (native/DESIGN.md 7.3), or the session is ending: waits for
    /// folder claims in flight, deletes the folders of runs still going, whose conversions
    /// end with the process, and applies the results of runs that ended before it, so
    /// they reach the list and History. After this nothing starts and later results change nothing.
    /// </summary>
    public void DiscardConverting()
    {
        List<string> folders;
        List<ConversionItem> finished;
        lock (gate)
        {
            quitting = true;
            var deadline = DateTime.UtcNow + ClaimWait;
            while (inFlight > 0 && deadline - DateTime.UtcNow is { Ticks: > 0 } left)
                Monitor.Wait(gate, left);
            folders = [.. claimed.Values];
            claimed.Clear();
            finished = [.. ended.Keys];
        }
        foreach (var folder in folders)
            Conversion.Delete(folder);
        foreach (var item in finished)
            Finish(item);
    }

    void Finish(ConversionItem item)
    {
        ItemState? state;
        lock (gate)
        {
            if (!ended.Remove(item, out state))
                return;
        }
        item.State = state;
        StateChanged?.Invoke();
        Finished?.Invoke(item);
        Pump();
    }
}

/// <summary>One conversion, start to end, off the UI thread (native/DESIGN.md 4.1, "Running an item").</summary>
public static class Conversion
{
    /// <summary>A claimed folder, or why there is none.</summary>
    public sealed record Claimed(string? Folder, Failed? Failure);

    /// <summary>Claims the item's folder in <paramref name="parent"/> (native/DESIGN.md 4.3).</summary>
    public static Claimed Claim(IConverter converter, Source source, string parent)
    {
        try
        {
            var claim = CoreJson.Parse(converter.ClaimOutputFolder(CoreJson.ClaimRequest(parent, source)));
            if (!claim.Ok)
                return new(null, new Failed("output_unavailable", claim.Message));
            return new(CoreJson.Text(claim.Result, "path") ?? throw new InvalidOperationException("The folder claim answered without a path."), null);
        }
        catch (Exception e)
        {
            return new(null, new Failed("internal_error", e.Message));
        }
    }

    /// <summary>Converts into a claimed folder; a failure deletes the folder.</summary>
    public static ItemState Convert(IConverter converter, Source source, string folder, ConversionOptions options)
    {
        ItemState state;
        try
        {
            state = CoreJson.Conversion(converter.ConvertJson(CoreJson.ConvertRequest(source, folder, options)), folder);
        }
        catch (Exception e)
        {
            state = new Failed("internal_error", e.Message);
        }
        if (state is Failed)
            Delete(folder);
        return state;
    }

    /// <summary>Deletes a folder this app claimed, and everything in it; a folder already gone is fine.</summary>
    public static void Delete(string folder)
    {
        try
        {
            Directory.Delete(folder, recursive: true);
        }
        catch (DirectoryNotFoundException)
        {
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
