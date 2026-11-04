package model

import (
	"slices"
	"time"
)

// Slots is how many items convert at a time.
const Slots = 2

// Queue is the Convert list: newest first, processed oldest first, with a
// single selection.
type Queue struct {
	Items []*Item
	// Selected is the selected item's ID, "" for none.
	Selected ItemID
	seq      uint64
}

// Add puts sources at the top in the order given and returns the
// duplicates refused: sources already Queued or Converting. The first item
// added to an empty list is selected.
func (q *Queue) Add(sources []Source, o Options, now time.Time) (added []*Item, duplicates []Source) {
	wasEmpty := len(q.Items) == 0
	for _, s := range sources {
		if q.waiting(s) || slices.ContainsFunc(added, func(it *Item) bool { return it.Source == s }) {
			duplicates = append(duplicates, s)
			continue
		}
		q.seq++
		added = append(added, &Item{ID: newID(), Source: s, Title: s.Title(), AddedAt: now, Options: o, State: Queued{}, seq: q.seq})
	}
	q.Items = append(slices.Clone(added), q.Items...)
	if wasEmpty && len(added) > 0 {
		q.Selected = added[0].ID
	}
	return added, duplicates
}

func (q *Queue) waiting(s Source) bool {
	return slices.ContainsFunc(q.Items, func(it *Item) bool {
		switch it.State.(type) {
		case Queued, Converting:
			return it.Source == s
		}
		return false
	})
}

// Item returns the item with id, or nil.
func (q *Queue) Item(id ItemID) *Item {
	if i := q.index(id); i >= 0 {
		return q.Items[i]
	}
	return nil
}

func (q *Queue) index(id ItemID) int {
	return slices.IndexFunc(q.Items, func(it *Item) bool { return it.ID == id })
}

// Count counts the items in state S.
func Count[S State](q *Queue) int {
	n := 0
	for _, it := range q.Items {
		if _, ok := it.State.(S); ok {
			n++
		}
	}
	return n
}

// AnyFinished reports whether Clear Finished has anything to clear.
func (q *Queue) AnyFinished() bool { return slices.ContainsFunc(q.Items, (*Item).Finished) }

// Claim creates an item's output folder: the folder's path, or the error's
// code and message.
type Claim func(source string) (folder, code, message string)

// Start moves the oldest Queued items to Converting while fewer than Slots
// convert, claiming each one's folder, and returns the jobs to run. A
// failed claim fails the item with output_unavailable and returns it in
// failed, to be announced.
func (q *Queue) Start(claim Claim, outputFolder string, now time.Time) (jobs []Job, failed []*Item) {
	for Count[Converting](q) < Slots {
		it := q.oldestQueued()
		if it == nil {
			break
		}
		folder, code, message := claim(it.Source.Value)
		if code != "" {
			it.State = Failed{Code: "output_unavailable", Message: message, Folder: outputFolder}
			failed = append(failed, it)
			continue
		}
		it.State = Converting{StartedAt: now, Folder: folder}
		jobs = append(jobs, Job{ID: it.ID, Folder: folder, Request: Request(it.Source, folder, it.Options)})
	}
	return jobs, failed
}

func (q *Queue) oldestQueued() *Item {
	var oldest *Item
	for _, it := range q.Items {
		if _, ok := it.State.(Queued); ok && (oldest == nil || it.seq < oldest.seq) {
			oldest = it
		}
	}
	return oldest
}

// Finish records a job's outcome. A Done URL takes its front matter's
// title. It returns the item, nil when it is gone.
func (q *Queue) Finish(id ItemID, s State) *Item {
	it := q.Item(id)
	if it == nil {
		return nil
	}
	it.State = s
	if d, ok := s.(Done); ok && it.Source.Kind == URL && d.Result.Title != "" {
		it.Title = d.Result.Title
	}
	return it
}

// Remove takes an item out of the list unless it is converting; the
// selection moves to the next row, else the previous one.
func (q *Queue) Remove(id ItemID) bool {
	i := q.index(id)
	if i < 0 {
		return false
	}
	if _, busy := q.Items[i].State.(Converting); busy {
		return false
	}
	q.Items = slices.Delete(q.Items, i, i+1)
	if q.Selected == id {
		q.Selected = ""
		switch {
		case i < len(q.Items):
			q.Selected = q.Items[i].ID
		case i > 0:
			q.Selected = q.Items[i-1].ID
		}
	}
	return true
}

// ClearFinished removes every Done and Failed item and returns their IDs.
func (q *Queue) ClearFinished() (removed []ItemID) {
	for _, it := range slices.Clone(q.Items) {
		if it.Finished() && q.Remove(it.ID) {
			removed = append(removed, it.ID)
		}
	}
	return removed
}

// Retry queues a Failed item again with the current options.
func (q *Queue) Retry(id ItemID, o Options) bool {
	it := q.Item(id)
	if it == nil {
		return false
	}
	if _, ok := it.State.(Failed); !ok {
		return false
	}
	q.seq++
	it.Options, it.State, it.seq = o, Queued{}, q.seq
	return true
}

// Move selects the row delta rows from the selected one.
func (q *Queue) Move(delta int) {
	if len(q.Items) == 0 {
		return
	}
	i := max(q.index(q.Selected), 0) + delta
	q.Selected = q.Items[min(max(i, 0), len(q.Items)-1)].ID
}

// ConvertingFolders lists the folders claimed for items still converting,
// which Quit deletes (7.3).
func (q *Queue) ConvertingFolders() []string {
	var folders []string
	for _, it := range q.Items {
		if c, ok := it.State.(Converting); ok {
			folders = append(folders, c.Folder)
		}
	}
	return folders
}
