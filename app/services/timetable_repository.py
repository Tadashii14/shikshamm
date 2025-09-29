"""
Timetable generation algorithm from the repository
"""
import pandas as pd
import datetime
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import math
import random

class PomodoroTimetableGenerator:
    def __init__(self):
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Study blocks from 6 AM to 9 PM (15 hours available)
        # Each block is 1 hour study + 15 min break = 1.25 hours
        self.study_start_hour = 6  # 6 AM
        self.study_end_hour = 21   # 9 PM (21:00)
        self.pomodoro_block_duration = 1.25  # 1 hour study + 15 min break
        
    def format_time(self, hour: int, minute: int = 0) -> str:
        """Convert 24-hour format to readable time"""
        total_minutes = hour * 60 + minute
        display_hour = total_minutes // 60
        display_minute = total_minutes % 60
        
        if display_hour == 0:
            return f"12:{display_minute:02d} AM"
        elif display_hour < 12:
            return f"{display_hour}:{display_minute:02d} AM"
        elif display_hour == 12:
            return f"12:{display_minute:02d} PM"
        else:
            return f"{display_hour-12}:{display_minute:02d} PM"
    
    def parse_time_ranges(self, time_input: str) -> List[float]:
        """Parse time ranges and convert to decimal hours"""
        busy_times = []
        if not time_input.strip():
            return busy_times
            
        try:
            ranges = time_input.split(',')
            for range_str in ranges:
                range_str = range_str.strip()
                if '-' in range_str:
                    start, end = range_str.split('-')
                    start_hour = float(start.strip())
                    end_hour = float(end.strip())
                    
                    # Add all 15-minute intervals in the range
                    current = start_hour
                    while current < end_hour:
                        busy_times.append(current)
                        current += 0.25  # 15-minute increments
                else:
                    # Single hour
                    busy_times.append(float(range_str))
        except ValueError:
            return []
        
        return busy_times
    
    def get_available_pomodoro_slots(self, busy_hours: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Find available Pomodoro time slots (1.25-hour blocks) for each day"""
        available_slots = {}
        
        for day in self.days:
            busy_today = set(busy_hours.get(day, []))
            available_blocks = []
            
            # Check each possible Pomodoro block start time
            current_time = self.study_start_hour
            while current_time + self.pomodoro_block_duration <= self.study_end_hour:
                # Check if this entire 1.25-hour block is free
                block_free = True
                check_time = current_time
                
                while check_time < current_time + self.pomodoro_block_duration:
                    if check_time in busy_today:
                        block_free = False
                        break
                    check_time += 0.25  # Check every 15 minutes
                
                if block_free:
                    available_blocks.append(current_time)
                
                current_time += self.pomodoro_block_duration  # Move to next possible block
            
            available_slots[day] = available_blocks
        
        return available_slots
    
    def calculate_subject_distribution(self, subjects: Dict[str, int], total_blocks_per_week: int) -> Dict[str, int]:
        """
        Distribute subjects evenly throughout the week based on difficulty
        """
        if not subjects:
            return {}
        
        # Calculate total difficulty points
        total_difficulty = sum(subjects.values())
        
        # Ensure each subject gets at least 1 block per week
        min_blocks_per_subject = 1
        base_blocks = len(subjects) * min_blocks_per_subject
        remaining_blocks = max(0, total_blocks_per_week - base_blocks)
        
        # Distribute remaining blocks based on difficulty
        allocation = {}
        for subject, difficulty in subjects.items():
            base_allocation = min_blocks_per_subject
            difficulty_bonus = int((difficulty / total_difficulty) * remaining_blocks)
            allocation[subject] = base_allocation + difficulty_bonus
        
        # Ensure we don't exceed total blocks
        total_allocated = sum(allocation.values())
        if total_allocated > total_blocks_per_week:
            # Reduce allocations proportionally
            reduction_factor = total_blocks_per_week / total_allocated
            allocation = {subject: max(1, int(blocks * reduction_factor)) 
                         for subject, blocks in allocation.items()}
        
        return allocation
    
    def create_pomodoro_timetable(self, subjects: Dict[str, int], busy_hours: Dict[str, List[float]], 
                                 max_blocks_per_day: int) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
        """
        Create Pomodoro-based timetable with even distribution
        """
        # Get available Pomodoro slots
        available_slots = self.get_available_pomodoro_slots(busy_hours)
        
        # Calculate total available blocks per week
        total_available_blocks = sum(min(len(slots), max_blocks_per_day) 
                                   for slots in available_slots.values())
        
        # Distribute subjects based on difficulty
        subject_allocation = self.calculate_subject_distribution(subjects, total_available_blocks)
        
        # Create timetable structure
        timetable = {}
        for day in self.days:
            timetable[day] = []
        
        # Track how many blocks each subject has been assigned
        assigned_blocks = {subject: 0 for subject in subjects.keys()}
        
        # Sort subjects by difficulty (hardest first) for better time slots
        sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)
        
        # Assign subjects to time slots with even distribution
        for day in self.days:
            day_slots = available_slots[day][:max_blocks_per_day]
            
            # Create a rotation of subjects for even distribution
            available_subjects = []
            for subject, difficulty in sorted_subjects:
                needed_blocks = subject_allocation[subject]
                if assigned_blocks[subject] < needed_blocks:
                    # Add subject multiple times based on how many blocks it needs
                    blocks_needed = needed_blocks - assigned_blocks[subject]
                    available_subjects.extend([subject] * min(blocks_needed, 3))  # Max 3 per day
            
            # Assign subjects to available slots for this day
            for i, slot_start in enumerate(day_slots):
                if i < len(available_subjects):
                    subject = available_subjects[i % len(available_subjects)]
                    
                    # Create the Pomodoro block
                    study_end = slot_start + 1  # 1 hour study
                    break_end = study_end + 0.25  # 15 minute break
                    
                    pomodoro_block = {
                        'subject': subject,
                        'study_start': slot_start,
                        'study_end': study_end,
                        'break_end': break_end,
                        'difficulty': subjects[subject]
                    }
                    
                    timetable[day].append(pomodoro_block)
                    assigned_blocks[subject] += 1
        
        return timetable, subject_allocation
    
    def create_schedule_visualization(self, timetable: Dict[str, List[Dict]], subjects: Dict[str, int]):
        """Create a detailed schedule visualization"""
        
        # Prepare data for the timeline chart
        schedule_data = []
        colors = px.colors.qualitative.Set3
        subject_colors = {subject: colors[i % len(colors)] for i, subject in enumerate(subjects.keys())}
        
        for day_idx, day in enumerate(self.days):
            day_schedule = timetable[day]
            
            for block in day_schedule:
                # Study block
                schedule_data.append({
                    'Day': day,
                    'Start': block['study_start'],
                    'End': block['study_end'],
                    'Subject': block['subject'],
                    'Type': 'Study',
                    'Color': subject_colors[block['subject']],
                    'Day_Num': day_idx
                })
                
                # Break block
                schedule_data.append({
                    'Day': day,
                    'Start': block['study_end'],
                    'End': block['break_end'],
                    'Subject': 'Break',
                    'Type': 'Break',
                    'Color': '#E8E8E8',
                    'Day_Num': day_idx
                })
        
        if not schedule_data:
            return None
            
        df = pd.DataFrame(schedule_data)
        
        # Create Gantt-style chart
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                name=row['Subject'],
                y=[row['Day']],
                x=[row['End'] - row['Start']],
                base=[row['Start']],
                orientation='h',
                marker_color=row['Color'],
                text=f"{row['Subject']}" if row['Type'] == 'Study' else 'Break',
                textposition='inside',
                showlegend=False,
                hovertemplate=f"<b>{row['Subject']}</b><br>" +
                             f"Time: {self.format_time(int(row['Start']), int((row['Start'] % 1) * 60))}-" +
                             f"{self.format_time(int(row['End']), int((row['End'] % 1) * 60))}<br>" +
                             f"Duration: {(row['End'] - row['Start']) * 60:.0f} minutes<extra></extra>"
            ))
        
        fig.update_layout(
            title="Weekly Pomodoro Study Schedule",
            xaxis_title="Time of Day",
            yaxis_title="Day",
            height=600,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(6, 22, 2)),
                ticktext=[self.format_time(h) for h in range(6, 22, 2)],
                range=[6, 21]
            ),
            barmode='stack'
        )
        
        return fig
    
    def create_daily_summary(self, timetable: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Create a summary table of daily schedules"""
        summary_data = []
        
        for day in self.days:
            day_blocks = timetable[day]
            if day_blocks:
                study_subjects = [block['subject'] for block in day_blocks]
                total_study_time = len(day_blocks)  # Each block is 1 hour
                start_time = min(block['study_start'] for block in day_blocks)
                end_time = max(block['break_end'] for block in day_blocks)
                
                summary_data.append({
                    'Day': day,
                    'Study Hours': total_study_time,
                    'Subjects': ', '.join(study_subjects),
                    'Start Time': self.format_time(int(start_time), int((start_time % 1) * 60)),
                    'End Time': self.format_time(int(end_time), int((end_time % 1) * 60))
                })
            else:
                summary_data.append({
                    'Day': day,
                    'Study Hours': 0,
                    'Subjects': 'Rest Day',
                    'Start Time': '-',
                    'End Time': '-'
                })
        
        return pd.DataFrame(summary_data)


def generate_timetable_from_repository(
    subjects: List[str],
    difficulty_levels: List[str],
    study_hours_per_day: int
) -> Dict:
    """
    Generates a Pomodoro-based study timetable using the repository's algorithm.
    """
    generator = PomodoroTimetableGenerator()

    # Map subjects to difficulty levels
    subjects_with_difficulty = {}
    difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
    for i, subject_name in enumerate(subjects):
        if i < len(difficulty_levels):
            subjects_with_difficulty[subject_name] = difficulty_map.get(difficulty_levels[i].lower(), 2) # Default to medium
        else:
            subjects_with_difficulty[subject_name] = 2 # Default to medium if no difficulty provided

    # Busy hours (assuming no busy hours for simplicity in this API integration)
    # In a full UI, this would come from user input
    busy_hours = {day: [] for day in generator.days}

    # Max blocks per day based on study_hours_per_day
    # Each block is 1 hour study + 15 min break = 1.25 hours
    max_blocks_per_day = math.floor(study_hours_per_day / 1.25)
    if max_blocks_per_day == 0 and study_hours_per_day > 0:
        max_blocks_per_day = 1 # Ensure at least one block if study hours are provided

    timetable, subject_allocation = generator.create_pomodoro_timetable(
        subjects_with_difficulty, busy_hours, max_blocks_per_day
    )

    # Format timetable for API response
    formatted_timetable = []
    for day, blocks in timetable.items():
        for block in blocks:
            formatted_timetable.append({
                "day": day,
                "subject": block['subject'],
                "difficulty": block['difficulty'],
                "start_time": generator.format_time(int(block['study_start']), int((block['study_start'] % 1) * 60)),
                "end_time": generator.format_time(int(block['study_end']), int((block['study_end'] % 1) * 60)),
                "break_end_time": generator.format_time(int(block['break_end']), int((block['break_end'] % 1) * 60))
            })
    
    # Create summary
    daily_summary_df = generator.create_daily_summary(timetable)
    daily_summary = daily_summary_df.to_dict('records')

    # Calculate metrics
    total_blocks = sum(len(blocks) for blocks in timetable.values())
    total_study_hours = total_blocks # Each block is 1 hour study
    total_time_with_breaks = total_blocks * 1.25
    
    return {
        "success": True,
        "schedule": formatted_timetable,
        "subject_allocation": subject_allocation,
        "daily_summary": daily_summary,
        "metrics": {
            "total_blocks": total_blocks,
            "total_study_hours": total_study_hours,
            "total_time_with_breaks": total_time_with_breaks,
            "daily_average_hours_with_breaks": round(total_time_with_breaks / 7, 2)
        }
    }