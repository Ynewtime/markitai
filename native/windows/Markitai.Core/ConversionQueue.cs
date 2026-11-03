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

    // The folders claimed for items still Converting. Workers add to it, the owner's
    // thread reads it when quitting, so it is the one piece of state under a lock.
    readonly Lock gate = new();
    readonly Dictionary<ConversionItem, string> claimed = [];
    bool quitting;

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
        while (ConvertingCount < Slots && Items.Where(i => i.State is Queued).MinBy(i => i.Order) is { } next)
            Start(next);
    }

    void Start(ConversionItem item)
    {
        item.State = new Converting(clock.GetUtcNow());
        StateChanged?.Invoke();
        var (source, parent, options) = (item.Source, outputFolder(), item.Options);
        Task.Run(() =>
        {
            var state = Conversion.Run(converter, source, parent, options, folder => Claimed(item, folder));
            post(() => Finish(item, state));
        });
    }

    void Claimed(ConversionItem item, string folder)
    {
        lock (gate)
        {
            if (!quitting)
            {
                claimed[item] = folder;
                return;
            }
        }
        Conversion.Delete(folder);
    }

    /// <summary>
    /// The user confirmed Quit (native/DESIGN.md 7.3): deletes the folders claimed for
    /// items still Converting, whose conversions end with the process. Conversions that
    /// come back after this change nothing, so nothing they wrote reaches History.
    /// </summary>
    public void DiscardConverting()
    {
        List<string> folders;
        lock (gate)
        {
            quitting = true;
            folders = [.. Items.Where(i => i.State is Converting).Select(i => claimed.GetValueOrDefault(i)).OfType<string>()];
        }
        foreach (var folder in folders)
            Conversion.Delete(folder);
    }

    void Finish(ConversionItem item, ItemState state)
    {
        lock (gate)
        {
            if (quitting)
                return;
            claimed.Remove(item);
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
    /// <param name="claimed">Told the folder as soon as it is claimed, before the conversion starts.</param>
    public static ItemState Run(IConverter converter, Source source, string parent, ConversionOptions options, Action<string>? claimed = null)
    {
        string folder;
        try
        {
            var claim = CoreJson.Parse(converter.ClaimOutputFolder(CoreJson.ClaimRequest(parent, source)));
            if (!claim.Ok)
                return new Failed("output_unavailable", claim.Message);
            folder = CoreJson.Text(claim.Result, "path") ?? throw new InvalidOperationException("The folder claim answered without a path.");
        }
        catch (Exception e)
        {
            return new Failed("internal_error", e.Message);
        }
        claimed?.Invoke(folder);
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
