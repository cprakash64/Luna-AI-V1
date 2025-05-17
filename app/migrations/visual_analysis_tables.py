"""
Database migrations for visual analysis tables
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic
revision = '7b2f04c9a8d1'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create frames table
    op.create_table(
        'frames',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('video_id', UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp', sa.Float, nullable=False),
        sa.Column('frame_path', sa.String, nullable=True),
        sa.Column('objects', JSONB, nullable=True),
        sa.Column('detected_objects', sa.String, nullable=True),
        sa.Column('text', sa.Text, nullable=True),
        sa.Column('ocr_text', sa.Text, nullable=True),
        sa.Column('scene_colors', JSONB, nullable=True),
        sa.Column('object_counts', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
    )
    
    # Add index on video_id and timestamp
    op.create_index('ix_frames_video_id_timestamp', 'frames', ['video_id', 'timestamp'])
    
    # Create scenes table
    op.create_table(
        'scenes',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('video_id', UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=False),
        sa.Column('start_time', sa.Float, nullable=False),
        sa.Column('end_time', sa.Float, nullable=False),
        sa.Column('duration', sa.Float, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('key_objects', JSONB, nullable=True),
        sa.Column('dominant_colors', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
    )
    
    # Add index on video_id and start_time
    op.create_index('ix_scenes_video_id_start_time', 'scenes', ['video_id', 'start_time'])
    
    # Create highlights table
    op.create_table(
        'highlights',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('video_id', UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp', sa.Float, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('objects', JSONB, nullable=True),
        sa.Column('is_scene_change', sa.Boolean, default=False, nullable=False),
        sa.Column('has_text', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
    )
    
    # Add index on video_id and timestamp
    op.create_index('ix_highlights_video_id_timestamp', 'highlights', ['video_id', 'timestamp'])
    
    # Create visual_summary table
    op.create_table(
        'visual_summary',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('video_id', UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('overall_summary', sa.Text, nullable=True),
        sa.Column('key_visual_elements', JSONB, nullable=True),
        sa.Column('color_palette', JSONB, nullable=True),
        sa.Column('visual_timeline', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
    )
    
    # Create index on video_id
    op.create_index('ix_visual_summary_video_id', 'visual_summary', ['video_id'])
    
    # Create topics table
    op.create_table(
        'topics',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('video_id', UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('timestamps', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('now()'), nullable=False),
    )
    
    # Add index on video_id
    op.create_index('ix_topics_video_id', 'topics', ['video_id'])
    
    # Add columns to videos table
    op.add_column('videos', sa.Column('visual_analysis_status', sa.String, nullable=True))
    op.add_column('videos', sa.Column('has_visual_analysis', sa.Boolean, server_default='false', nullable=False))
    op.add_column('videos', sa.Column('visual_analysis_completed_at', sa.DateTime, nullable=True))

def downgrade():
    # Drop columns from videos table
    op.drop_column('videos', 'visual_analysis_completed_at')
    op.drop_column('videos', 'has_visual_analysis')
    op.drop_column('videos', 'visual_analysis_status')
    
    # Drop topics table
    op.drop_table('topics')
    
    # Drop visual_summary table
    op.drop_table('visual_summary')
    
    # Drop highlights table
    op.drop_table('highlights')
    
    # Drop scenes table
    op.drop_table('scenes')
    
    # Drop frames table
    op.drop_table('frames')